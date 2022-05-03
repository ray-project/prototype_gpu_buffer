import os
import subprocess
import shutil
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import tensorflow as tf


class BuildTensorTransfer(BuildExtension):
    def run(self):
        super().run()
        compile_command = ["g++", "-std=c++11", "-shared", "src/tensorflow_bind.cpp", "src/gpu_comm.cpp",
                           "-I/usr/local/cuda/include", "-o", "tensortransfer/core/tensorflow_bind.so", "-fPIC", "-O2"]
        compile_command.extend(tf.sysconfig.get_compile_flags())
        compile_command.extend(tf.sysconfig.get_link_flags())
        os.makedirs("tensortransfer/core/", exist_ok=True)
        subprocess.check_call(compile_command)
        target_dir = os.path.join(self.build_lib, "tensortransfer", "core")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy("tensortransfer/core/tensorflow_bind.so", target_dir)

setup(
    name='tensortransfer',
    ext_modules=[
        CUDAExtension('tensortransfer_torch_ext', sources=[
            # NOTE: torch extension will somehow ignore file extension, which will cause problems.
            'src/torch_bind.cpp',
            'src/gpu_comm.cpp',
        ], extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}),
    ],
    cmdclass={
        'build_ext': BuildTensorTransfer
    },
    packages=find_packages()
)
