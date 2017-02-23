# functions/add.py
import torch
from torch.autograd import Function
import numpy as np

class GridGenFunction(Function):
    def __init__(self, height, width,lr=1):
        super(GridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print self.grid

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if not input1.is_cuda:
            for i in range(input1.size(0)):
                #output[i,:,:,0] = self.grid[:,:,0] * input1[i,0,0] + self.grid[:,:,1] * input1[i,0,1] + input1[i,0,2]
                #output[i,:,:,1] = self.grid[:,:,1] * input1[i,1,0] + self.grid[:,:,1] * input1[i,1,1] + input1[i,1,2]
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)
        else:
            print 'not implemented'
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        if not grad_output.is_cuda:
            #print 'gradout:',grad_output.size()
            grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))
            #print grad_input1.size()
        else:
            print 'not implemented'
        return grad_input1 * self.lr
