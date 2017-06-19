from torch.nn.modules.module import Module
from functions.stn import STNFunction, STNFunctionBCHW

class STN(Module):
    def __init__(self, layout = 'BHWD'):
        super(STN, self).__init__()
        if layout == 'BHWD':
            self.f = STNFunction()
        else:
            self.f = STNFunctionBCHW()
    def forward(self, input1, input2):
        return self.f(input1, input2)
