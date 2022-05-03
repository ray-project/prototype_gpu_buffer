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

        @self.app.route("/<object_id>")
        def fetch(object_id):
            try:
                return f"{cache[int(object_id)]}"
            except:
                return f"Key not found"

    def init(self):
        self.app.run(port=self.port)

    def _fetch(self, obj):
        if obj.obj_id not in cache:
            url = f"http://127.0.0.1:{obj.src}/{obj.obj_id}"
            cache[obj.obj_id] = requests.get(url).content
        return cache[obj.obj_id]

    def _put(self, obj):
        obj_id = self.obj_id
        self.obj_id += 1

        cache[obj_id] = obj
        return HttpObject(obj_id, src=self.port)


@ray.remote(max_concurrency=2)
class Worker:
    def __init__(self, port):
        self.object_store = HttpObjectStore(port)

    # Wrapper methods to use HTTP for transport.
    def init(self):
        self.object_store.init()

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
    w1.init.remote()
    w2 = Worker.remote(5001)
    w2.init.remote()

    print(ray.get(w2.recv.remote(w1.ping.remote())))
    print(ray.get(w2.recv.remote(w1.ping.remote(use_http=True))))
