from torch.nn.modules.module import Module
from functions.stn_invert import STNInvertFunction

class STNInvert(Module):
    def __init__(self):
        super(STNInvert, self).__init__()
        self.f = STNInvertFunction()
    def forward(self, input1, input2, input3 = None):
        if input3 is None:
            return self.f(input1, input2)
        else:
            return self.f(input1, input2, input3)