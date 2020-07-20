import torch
import torch.nn as nn

class baseDCNNLoss(nn.Module):
    def __init__(self):
        super(baseDCNNLoss,self).__init__()
        self.loss=nn.CrossEntropyLoss()
    def forward(self,x,y):
        """
        x: torch.tensor sized [bsz,#class]
        y: torch.tensor sized [bsz]
        """
        return self.loss(x,y)