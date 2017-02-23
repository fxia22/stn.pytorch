from torch.nn.modules.module import Module
from functions.stn import STNFunction

class STN(Module):
    def __init__(self):
        super(STN, self).__init__()
        self.f = STNFunction()
    def forward(self, input1, input2):
        return self.f(input1, input2)
