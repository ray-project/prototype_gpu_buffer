import base64
import uuid
import threading
from typing import (
    Callable,
    List,
)
import cupy as cp
import ray
import ray.util.collective as collective
from ray.actor import ActorHandle
from flask import Flask
import requests
import pickle
import urllib.parse

from gpu_object_ref import GpuObjectRef

GROUP_NAME_PREFIX = "__ray_gpu_store"


@ray.remote(num_cpus=0)
def _send_request(url: str):
    requests.get(url)


class CollectiveCoordinator:
    """Using HTTP Server to coordinate collective calls between participants."""

    def __init__(
        self,
        send_fn: Callable[[GpuObjectRef, int], None],
        recv_fn: Callable[[GpuObjectRef, int], None],
    ):
        import logging

        # disable flask info logs
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        self.app = Flask(__name__)
        self.send_fn = send_fn
        self.recv_fn = recv_fn

        @self.app.route("/send")
        def send():
            from flask import request

            dst = request.args.get("dst")
            ref = request.args.get("ref")
            self.send_fn(pickle.loads(base64.b64decode(ref)), int(dst))
            return "OK"

        @self.app.route("/recv")
        def recv():
            from flask import request

            src = request.args.get("src")
            ref = request.args.get("ref")
            self.recv_fn(pickle.loads(base64.b64decode(ref)), int(src))
            return "OK"

    def _run(self):
        from werkzeug import serving

        self.server = serving.make_server(
            host=ray.util.get_node_ip_address(), port=0, app=self.app, threaded=False
        )
        self.port = self.server.socket.getsockname()[1]
        self.server.serve_forever()

    def run(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def ray_call_send(address: str, dst_rank: int, ref: GpuObjectRef):
        serialized = urllib.parse.quote(
            base64.b64encode(pickle.dumps(ref)).decode("utf-8"), safe=""
        )
        send_url = f"http://{address}/send?dst={dst_rank}&ref={serialized}"
        return _send_request.remote(send_url)

    @staticmethod
    def ray_call_recv(address: str, src_rank: int, ref: GpuObjectRef):
        serialized = urllib.parse.quote(
            base64.b64encode(pickle.dumps(ref)).decode("utf-8"), safe=""
        )
        send_url = f"http://{address}/recv?src={src_rank}&ref={serialized}"
        return _send_request.remote(send_url)


class _ActorGroup:
    """Represents an actor transfer group for GPU objects transfer"""

    def __init__(self, group_name: str, actors: List[ActorHandle]):
        self.group_name = group_name
        self.actors = actors
        self.actor_addresses = ray.get(
            [actor.get_coordinator_address.remote() for actor in actors]
        )


@ray.remote(num_gpus=0, num_cpus=0)
class GpuTransferManager:
    """Singleton transfer manager that manages all the actor transfer groups"""

    def __init__(self):
        self.group_idx = 0
        self.actor_groups = dict()

    def _get_new_group_name(self):
        self.group_idx += 1
        return f"{GROUP_NAME_PREFIX}:{self.group_idx}"

    def setup_transfer_group(self, actors: List[ActorHandle]):
        group_name = self._get_new_group_name()
        print(f"setup group {group_name} with actors {actors}")
        self.actor_groups[group_name] = _ActorGroup(group_name, actors)

        ranks = list(range(len(actors)))
        _options = {
            "group_name": group_name,
            "world_size": len(actors),
            "ranks": ranks,
            "backend": "nccl",
        }
        collective.create_collective_group(actors, **_options)

        ray.get(
            [
                actor._setup_transfer_group.remote(len(actors), rank, group_name)
                for rank, actor in enumerate(actors)
            ]
        )

    def collective_p2p_transfer(
        self, group_name: str, ref: GpuObjectRef, src: int, dst: int
    ):
        ray.get(
            [
                CollectiveCoordinator.ray_call_send(
                    self.actor_groups[group_name].actor_addresses[src], dst, ref
                ),
                CollectiveCoordinator.ray_call_recv(
                    self.actor_groups[group_name].actor_addresses[dst], src, ref
                ),
            ]
        )


def get_transfer_manager() -> "ActorHandle":
    return GpuTransferManager.options(
        name="_ray_gpu_transfer_manager",
        namespace="_RAY_GPU",
        lifetime="detached",
        get_if_exists=True,
    ).remote()


def setup_transfer_group(actors: List[ActorHandle]):
    transfer_manager = get_transfer_manager()
    ray.get(transfer_manager.setup_transfer_group.remote(actors))


class GpuActorBase:
    """Base actor class for GPU transfer."""

    def __init__(self):
        self.buffers = {}
        self.transfer_manager = None
        self.coordinator = CollectiveCoordinator(self.send_buffer, self.recv_buffer)
        self.coordinator.run()
        self.group_name = None

    def _setup_transfer_group(self, world_size: int, rank: int, group_name: str):
        assert self.group_name is None, "Gpu actor could only belong to one group."
        self.world_size = world_size
        self.rank = rank
        self.group_name = group_name

    def get_coordinator_address(self) -> str:
        return f"{ray.util.get_node_ip_address()}:{self.coordinator.port}"

    def put_gpu_buffer(self, buffer) -> GpuObjectRef:
        buffer_id = uuid.uuid4()
        self.buffers[buffer_id] = buffer
        return GpuObjectRef(
            buffer_id, self.group_name, self.rank, buffer.shape, buffer.dtype
        )

    def _get_gpu_buffer(self, ref: GpuObjectRef):
        assert self.contains(ref)
        return self.buffers[ref.id]

    def _get_transfer_manager(self) -> "ActorHandle":
        if not self.transfer_manager:
            self.transfer_manager = get_transfer_manager()
        return self.transfer_manager

    def get_gpu_buffer(self, ref: GpuObjectRef):
        assert (
            self.group_name == ref.group
        ), f"{self.group_name} is different from {ref.group}"
        if self.contains(ref):
            return self._get_gpu_buffer(ref)
        ray.get(
            self._get_transfer_manager().collective_p2p_transfer.remote(
                self.group_name, ref, ref.src_rank, self.rank
            )
        )
        return self._get_gpu_buffer(ref)

    def contains(self, ref: GpuObjectRef) -> bool:
        return ref.id in self.buffers

    def send_buffer(self, ref: GpuObjectRef, dest_rank: int) -> None:
        assert self.contains(ref)
        collective.send(self.buffers[ref.id], dest_rank, self.group_name)

    def recv_buffer(self, ref: GpuObjectRef, src_rank: int):
        assert not self.contains(ref)
        self.buffers[ref.id] = cp.ndarray(shape=ref.shape, dtype=ref.dtype)
        collective.recv(self.buffers[ref.id], src_rank, self.group_name)

    # TODO: support GC objects


@ray.remote(num_gpus=1)
class GpuActor(GpuActorBase):
    """Example class for gpu transfer."""

    def put_gpu_obj(self):
        object = cp.ones((1024 * 1024 * 100,), dtype=cp.float32)
        return self.put_gpu_buffer(object)

    def load_gpu_obj(self, tensor_ref: GpuObjectRef):
        buffer = self.get_gpu_buffer(tensor_ref)
        print(f"buffer received! {buffer}")


if __name__ == "__main__":
    sender_actor = GpuActor.options(num_gpus=1).remote()
    receiver_actor = GpuActor.options(num_gpus=1).remote()
    setup_transfer_group([sender_actor, receiver_actor])

    for _ in range(10):
        ref = sender_actor.put_gpu_obj.remote()
        ray.get(receiver_actor.load_gpu_obj.remote(ref))
