from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from functions.gridgen import AffineGridGenFunction, CylinderGridGenFunction


class AffineGridGen(Module):
    def __init__(self, height, width, lr = 1):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
    def forward(self, input):
        return self.f(input)



class CylinderGridGen(Module):
    def __init__(self, height, width, lr = 1):
        super(CylinderGridGen, self).__init__()
        self.height, self.width = height, width
        self.f = CylinderGridGenFunction(self.height, self.width, lr=lr)
    def forward(self, input):
        return self.f(input)
