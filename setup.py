from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


nvjpeg_found = (
    CUDA_HOME is not None and
      os.path.exists(os.path.join(CUDA_HOME, 'include', 'nvjpeg.h') and
      os.path.exists(os.path.join(CUDA_HOME, 'include', 'nvjpeg2k.h')))
)

print('NVJPEG found: {0}'.format(nvjpeg_found))


setup(
    name='nvjpeg_torch',
    version="0.1.1",
    packages=['nvjpeg_torch'],

    ext_modules=[
        CUDAExtension('nvjpeg_cuda', 
          [ 'nvjpeg_cuda.cpp' ],
          libraries=['nvjpeg']) ,
        CUDAExtension('nvjpeg2k_cuda', 
          [ 'nvjpeg2k_cuda.cpp' ],
          libraries=['nvjpeg2k']) ,
    ],


    cmdclass={
        'build_ext': BuildExtension
    })
