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
