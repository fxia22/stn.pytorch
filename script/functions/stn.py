# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib
from cffi import FFI
ffi = FFI()

class STNFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        #print('decice %d' % torch.cuda.current_device())
        self.device = torch.cuda.current_device()
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            output = output.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            my_lib.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2
