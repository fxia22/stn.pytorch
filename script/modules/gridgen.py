from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from functions.gridgen import AffineGridGenFunction, CylinderGridGenFunction


class AffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1,0,0], [0,1,0]], dtype=np.float32))
            batch_identity = torch.zeros([input.size(0), 2,3])
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss,1)
            loss = torch.sum(loss,2)

            return self.f(input), loss.view(-1,1)

class CylinderGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):
        
        if not self.aux_loss:
            return self.f(input)
        else:
            return self.f(input), torch.mul(input, input).view(-1,1)

        
class AffineGridGenV2(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        
        
    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)
        
        return output
    
    
class CylinderGridGenV2(Module):
    def __init__(self, height, width, lr = 1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.grid.size() )
        #print(self.batchgrid.size())
        for i in range(input.size(0)):
            self.batchgrid[i,:,:,:] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        
        #print(self.batchgrid.size())
        
        input_u = input.view(-1,1,1,1).repeat(1,self.height, self.width,1)
        #print(input_u.requires_grad, self.batchgrid)
        output = Variable(torch.zeros(torch.Size([input.size(0)]) + self.grid.size()[:2] + torch.Size([2])), requires_grad = True)
        for i in range(input.size(0)):
            output[i,:,:,0] = self.batchgrid[i,:,:,0]
            output[i,:,:,1] = torch.atan(torch.tan(np.pi/2.0*(self.batchgrid[i,:,:,1] + self.batchgrid[i,:,:,2] * input_u[i,:,:,:])))  /(np.pi/2)
        
        return output


class DenseAffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        
        
    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())

        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        
        self.batchgrid = Variable(self.batchgrid)
        #print self.batchgrid,  input1[:,:,:,0:3]
        #print self.batchgrid,  input1[:,:,:,4:6]
        x = torch.mul(self.batchgrid, input1[:,:,:,0:3])
        y = torch.mul(self.batchgrid, input1[:,:,:,3:6])
        
        output = torch.cat([torch.sum(x,3),torch.sum(y,3)], 3)
        return output
    
    

