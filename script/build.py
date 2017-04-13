import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/my_lib.c', 'src/my_lib_invert.c']
headers = ['src/my_lib.h', 'src/my_lib_invert.h']
defines = []
with_cuda = False


ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
