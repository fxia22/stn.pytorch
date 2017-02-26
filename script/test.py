import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from modules.stn import STN
from modules.gridgen import AffineGridGen, CylinderGridGen

nframes = 64
height = 64
width = 128
channels = 64

inputImages = torch.zeros(nframes, height, width, channels)
grids = torch.zeros(nframes, height, width, 2)

input1, input2 = Variable(inputImages, requires_grad=True), Variable(grids, requires_grad=True)

input1.data.uniform_()
input2.data.uniform_(-1,1)

input = Variable(torch.from_numpy(np.array([[[0.8, 0.3, 1], [0.5, 0, 0]]], dtype=np.float32)), requires_grad = True)
print input

g = AffineGridGen(64, 128, aux_loss = True)
out, aux = g(input)
print out.size()
print out
out.backward(out.data)
print input.grad



#print input2.data
s = STN()
out = s(input1, input2)
#print out
out.backward(input1.data)
print input1.grad.size()


input = Variable(torch.from_numpy(np.array([[3.6]], dtype=np.float32)), requires_grad = True)
print input

g = CylinderGridGen(64, 128, aux_loss = True)
out, aux = g(input)
print out.size()
print out
