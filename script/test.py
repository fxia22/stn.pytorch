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

with torch.cuda.device(3):
    input1 = input1.cuda()
    input2 = input2.cuda()
    start = time.time()
    out = s(input1, input2)
    print(out.size(), 'time:', time.time() - start)
    start = time.time()
    out.backward(input1.data.cuda())
    print('time:', time.time() - start)

s2 = STN(layout = 'BCHW')
input1, input2 = Variable(inputImages.transpose(2,3).transpose(1,2), requires_grad=True), Variable(grids.transpose(2,3).transpose(1,2), requires_grad=True)
input1.data.uniform_()
input2.data.uniform_(-1,1)
start = time.time()
out = s2(input1, input2)
print(out.size(), 'time:', time.time() - start)
start = time.time()
out.backward(input1.data)
print(input1.grad.size(), 'time:', time.time() - start)

with torch.cuda.device(1):
    input1 = input1.cuda()
    input2 = input2.cuda()
    start = time.time()
    out = s2(input1, input2)
    print(out.size(), 'time:', time.time() - start)
    start = time.time()
    out.backward(input1.data.cuda())
    print('time:', time.time() - start)

