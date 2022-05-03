import ray
from flask import Flask
import requests
import threading
import time


cache = {}

class HttpObject:
    def __init__(self, obj_id, src):
        self.obj_id = obj_id
        self.src = src

class HttpObjectStore:
    def __init__(self, port):
        self.port = port
        self.app = Flask(__name__)
        self.obj_id = 0

        @self.app.route("/put/<object_id>/<obj>")
        def put(object_id, obj):
            cache[int(object_id)] = obj
            return "OK"

    def init(self, self_handle):
        self.self_handle = self_handle
        self.app.run(port=self.port)

    def _fetch(self, obj):
        ray.get(obj.src.pull.remote(obj.obj_id, self.port))
        return cache[obj.obj_id]

    def _push(self, obj_id, dest):
        url = f"http://127.0.0.1:{dest}/put/{obj_id}/{cache[int(obj_id)]}"
        return requests.get(url)

    def _put(self, obj):
        obj_id = self.obj_id
        self.obj_id += 1

        cache[obj_id] = obj
        return HttpObject(obj_id, src=self.self_handle)


@ray.remote(max_concurrency=2)
class Worker:
    def __init__(self, port):
        self.object_store = HttpObjectStore(port)

    # Wrapper methods to use HTTP for transport.
    def init(self, self_handle):
        self.object_store.init(self_handle)

    def pull(self, obj_id, dest):
        return self.object_store._push(obj_id, dest)

    # "Actual" actor methods.
    def ping(self, use_http=False):
        val = "pong"
        if use_http:
            return self.object_store._put(val)
        else:
            return val

    def recv(self, reply):
        if isinstance(reply, HttpObject):
            return self.object_store._fetch(reply)
        else:
            return reply


if __name__ == "__main__":
    w1 = Worker.remote(5000)
    w1.init.remote(w1)
    w2 = Worker.remote(5001)
    w2.init.remote(w2)

    print(ray.get(w2.recv.remote(w1.ping.remote())))
    print(ray.get(w2.recv.remote(w1.ping.remote(use_http=True))))
