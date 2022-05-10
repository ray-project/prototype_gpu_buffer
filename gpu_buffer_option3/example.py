import cupy as cp
import ray
#from gpu_object_ref import GpuObjectRef
#from gpu_object_manager import GpuActorBase, setup_transfer_group
from mock import GpuObjectRef
from mock import GpuActorBase, setup_transfer_group

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

