
# coding: utf-8

# In[1]:

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from modules.stn_invert import STNInvert
from modules.gridgen import CylinderGridGen, AffineGridGen
from PIL import Image
import matplotlib.pyplot as plt



img = Image.open('cat.jpg').convert('RGB')
img = np.array(img)/255.0


# In[3]:

img_batch = np.expand_dims(img, 0)
inputImages = torch.from_numpy(img_batch.astype(np.float32))
inputImages.size()
s = STNInvert()
g = AffineGridGen(328, 582)
input = Variable(torch.from_numpy(np.array([[[1, 0.2, 0], [0.5, 1, 0]]], dtype=np.float32)), requires_grad = True)
#print input
out = g(input)
input1 = Variable(inputImages)


# In[9]:

res = s(input1, out)


# In[10]:

res.size()


# In[6]:



# In[11]:

res.backward(torch.rand(res.data.size()))

print(input.grad)

# In[ ]:



