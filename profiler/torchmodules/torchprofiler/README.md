# Profiling tool for PyTorch

Downloaded from [https://github.com/zhuwenxi/pytorch-profiling-tool](https://github.com/zhuwenxi/pytorch-profiling-tool).

### Prerequisites
* To enable this profiling tool, some changes have to be made to the exising PyTorch code. A Python interface has to be created for the "register_pre_hook()" function to access the backward pass' pre-hook, for both `THPCppFunction` and `THPFunction`. The implementation of `register_hook()` in `python_cpp_function.h` and `python_function.h" are relevant.
* A version of PyTorch with `register_pre_hook()` is available in the `pytorch-dev` docker container.
