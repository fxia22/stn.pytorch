from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from functions.gridgen import GridGenFunction


class GridGen(Module):
    def __init__(self, height, width):
        super(GridGen, self).__init__()
        self.height, self.width = height, width
        self.f = GridGenFunction(self.height, self.width)
        print self.f
    def forward(self, input):
        return self.f(input)
