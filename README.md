# PyTorch C FFI examples

In this repository you can find examples showing how to extend PyTorch with
custom C code. To use the ffi you need to install the `cffi` package from pip.

Currently there are two examples:
* `package` - a pip distributable package
* `script` - compiles the code into a local module, that can be later imported
    from other files
