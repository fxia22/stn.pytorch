# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib


class STNInvertFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.invgrid = torch.zeros(input2.size())
        output = torch.zeros(input1.size())
        if not input1.is_cuda:
            my_lib.InvSamplerBHWD_updateOutput(input1, input2, self.invgrid, output)
        else:
            print 'not implemented'
        return output#, self.invgrid

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        if not grad_output.is_cuda:
            my_lib.InvSamplerBHWD_updateGradInput(self.input1, self.input2, self.invgrid, grad_input1, grad_input2, grad_output)
        else:
            print 'not implemented'
        return grad_input1, grad_input2
