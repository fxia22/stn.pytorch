# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib
from cffi import FFI
ffi = FFI()
import time

class STNInvertFunction(Function):
    def forward(self, input1, input2, depth_map):
        #self.input1 = input1
        #self.input2 = input2
        if input1.is_cuda:
            invgrid = torch.zeros(input2.size()).cuda()
            output = torch.zeros(input1.size()).cuda()
        else:
            invgrid = torch.zeros(input2.size())
            output = torch.zeros(input1.size())
        
        
        self.device = torch.cuda.current_device() 
        self.device_c = ffi.new("int *")
        self.device_c[0] = self.device
        print(self.device_c[0])
                
        if not input1.is_cuda:
            my_lib.InvSamplerBHWD_updateOutput(input1, input2, invgrid, output, depth_map)
        else:
            self.target_depth_map = torch.zeros(depth_map.size()).cuda(self.device)
            start_time = time.time()
            my_lib.InvSamplerBHWD_updateOutput_cuda(input1, input2, invgrid, output, depth_map, self.target_depth_map, self.device_c)
            print("--- %s seconds ---" % (time.time() - start_time))
        
        self.save_for_backward(input1, input2, depth_map, invgrid)
        return output, invgrid

    def backward(self, grad_output, grad_invgrid):

        input1, input2, depth_map, invgrid = self.saved_tensors
        if not grad_output.is_cuda:
            grad_input1 = torch.zeros(input1.size())
            grad_input2 = torch.zeros(input2.size())
        else:
            grad_input1 = torch.zeros(input1.size()).cuda()
            grad_input2 = torch.zeros(input2.size()).cuda()
        
        self.device_c = ffi.new("int *")
        self.device_c[0] = self.device
        print(self.device_c[0])
        
        if not grad_output.is_cuda:
            my_lib.InvSamplerBHWD_updateGradInput(input1, input2, invgrid, grad_input1, grad_input2, grad_output)
        else:
            start_time = time.time()
            my_lib.InvSamplerBHWD_updateGradInput_cuda(input1, input2, invgrid, grad_input1, grad_input2, grad_output, self.device_c)
            print("--- %s seconds ---" % (time.time() - start_time))
            
        return grad_input1, grad_input2, depth_map