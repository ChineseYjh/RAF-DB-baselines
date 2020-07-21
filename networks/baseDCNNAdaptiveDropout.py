from networks.layers import *

class baseDCNNAdaptiveDropout(nn.Module):
    """
    return unnormalized tensor [bsz,7]
    """
    def __init__(self):
        super(baseDCNNAdaptiveDropout,self).__init__()
        self.DROPOUT1=0.5
        self.DROPOUT2=0.4
        self.DROPOUT3=2/7
        net=[ConvLayer(3,64,3,1,1,2,2,0)]
        net+=[ConvLayer(64,96,3,1,1,2,2,0)]
        net+=[ConvBlock(96,128,3,1,1)]
        net+=[ConvLayer(128,128,3,1,1,2,2,0)]
        net+=[ConvBlock(128,256,3,1,1)]
        net+=[ConvBlock(256,256,3,1,1)]
        self.convnet=nn.Sequential(*net)
        net=[nn.Dropout(p=self.DROPOUT1)]
        net+=[nn.Linear(256*12*12,2000)]
        net+=[nn.ReLU()]
        net+=[nn.Dropout(p=self.DROPOUT2)]
        net+=[nn.Linear(2000,7)]
        net+=[nn.Dropout(p=self.DROPOUT3)]
        self.fc=nn.Sequential(*net)
    def forward(self,x):
        y=self.convnet(x)
        y=y.flatten(start_dim=1) #[bsz,256*12*12]
        y=self.fc(y)
        return y