from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from modules.stn import STN
from modules.gridgen import AffineGridGen, CylinderGridGen, CylinderGridGenV2, DenseAffine3DGridGen, DenseAffine3DGridGen_rotate

import time

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
print(input)

g = AffineGridGen(64, 128, aux_loss = True)
out, aux = g(input)
print((out.size()))
out.backward(out.data)
print(input.grad.size())



#print input2.data
s = STN()
start = time.time()
out = s(input1, input2)
print(out.size(), 'time:', time.time() - start)

start = time.time()
out.backward(input1.data)
print(input1.grad.size(), 'time:', time.time() - start)

input1 = input1.cuda()
input2 = input2.cuda()

start = time.time()
out = s(input1, input2)
print(out.size(), 'time:', time.time() - start)
start = time.time()
out.backward(input1.data)
print('time:', time.time() - start)

input = Variable(torch.from_numpy(np.array([[3.6]], dtype=np.float32)), requires_grad = True)

g = CylinderGridGenV2(64, 128)
g2 = DenseAffine3DGridGen(64,128)
g3 = DenseAffine3DGridGen_rotate(64,128)
iden = torch.from_numpy(np.array([[1,0,0,-0.1],[0,1,0,-0.1],[0,0,1,0]]).astype(np.float32))
iden2 = iden.view(1,1,1,12).repeat(1,64,128,1)
iden2 = Variable(iden2, requires_grad = True)
input = input.repeat(16,1,1,1)
iden2 = iden2.repeat(16,1,1,1)

out = g(input)
print(out.size())
out.backward(torch.rand(16,64,128,2))
out = g2(iden2)
out.backward(torch.rand(16,64,128,2))
out = g3(iden2, input)
out.backward(torch.rand(16,64,128,2))
