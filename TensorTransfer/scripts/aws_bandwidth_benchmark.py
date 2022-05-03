import ray
import time
from utils import get_node_ip_address
import os


@ray.remote(num_gpus=1)
class RemoteActor:
    def __init__(self):
        os.system("sudo kill -9 `sudo lsof /var/lib/dpkg/lock-frontend | awk '{print $2}' | tail -n 1`")
        os.system("sudo pkill -9 apt-get")
        os.system("sudo pkill -9 dpkg")
        os.system("sudo kill -9 `sudo lsof /var/lib/dpkg/lock-frontend | awk '{print $2}' | tail -n 1`")
        os.system("sudo dpkg --configure -a")
        os.system("sudo apt-get install -y iperf")

    def start_service(self):
        os.system("iperf -s")

    def address(self):
        return get_node_ip_address()

    def benchmark(self, address):
        os.system("iperf -c " + address)


if __name__ == '__main__':
    ray.init(address='auto')
    service = RemoteActor.remote()
    client = RemoteActor.remote()
    address = ray.get(service.address.remote())
    service.start_service.remote()
    time.sleep(5)  # wait for service to start
    ray.get(client.benchmark.remote(address))
