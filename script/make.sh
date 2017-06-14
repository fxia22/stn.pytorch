#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o my_lib_cuda_kernel.cu.o my_lib_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52,sm_61
nvcc -c -o my_lib_invert_cuda_kernel.cu.o my_lib_invert_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52,sm_61

cd ../
python build.py
