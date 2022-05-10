Easy setup:
1. start ec2 instances with `Deep Learning AMI (Ubuntu 18.04) Version 60.1` image and `g3.16xlarge` machine type
2. log onto ec2 instance and run `source activate python3` to use ` Python3 (CUDA 11.0)` env
3. install ray nightly
4. install cupy by running `pip install cupy-cuda110`
5. run the example.py