# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib
from cffi import FFI
ffi = FFI()

class STNInvertFunction(Function):
    def forward(self, input1, input2, depth_map = None):
        #self.input1 = input1
        #self.input2 = input2
        if input1.is_cuda:
            self.invgrid = torch.zeros(input2.size()).cuda()
        else:
            self.invgrid = torch.zeros(input2.size())
        output = torch.zeros(input1.size())
        self.device = torch.cuda.current_device() 
        self.device_c = ffi.new("int *")
        self.device_c[0] = self.device
        print(self.device_c[0])
        
        self.depth_map = None
        if depth_map is None:
            self.depth_map = torch.zeros(input1.size()[0], input1.size()[1], input1.size()[2], 1)
            self.nodepth = True
        else:
            self.depth_map = depth_map
            self.nodepth = False
        
        if not input1.is_cuda:
            my_lib.InvSamplerBHWD_updateOutput(input1, input2, self.invgrid, output, self.depth_map)
        else:
            self.depth_map = self.depth_map.cuda(self.device)
            self.target_depth_map = torch.zeros(self.depth_map.size()).cuda(self.device)
            output = output.cuda(self.device)
            my_lib.InvSamplerBHWD_updateOutput_cuda(input1, input2, self.invgrid, output, self.depth_map, self.target_depth_map, self.device_c)
            
        self.save_for_backward(input1, input2)
        return output#, self.invgrid

    def backward(self, grad_output):

        input1, input2 = self.saved_tensors
        grad_input1 = torch.zeros(input1.size())
        grad_input2 = torch.zeros(input2.size())
        
        self.device_c = ffi.new("int *")
        self.device_c[0] = self.device
        print(self.device_c[0])
        
        if not grad_output.is_cuda:
            my_lib.InvSamplerBHWD_updateGradInput(input1, input2, self.invgrid, grad_input1, grad_input2, grad_output)
        else:
            #from IPython import embed; embed()
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib.InvSamplerBHWD_updateGradInput_cuda(input1, input2, self.invgrid, grad_input1, grad_input2, grad_output, self.device_c)
            
        if self.nodepth:
            return grad_input1, grad_input2
        else:
            if not grad_output.is_cuda:
                return grad_input1, grad_input2, torch.zeros(self.depth_map.size())
            else:
                return grad_input1, grad_input2, torch.zeros(self.depth_map.size()).cuda(self.device)
