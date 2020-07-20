from networks.layers import *

class baseDCNN_dropout(nn.Module):
    """
    return unnormalized tensor [bsz,7]
    """
    def __init__(self):
        super(baseDCNN,self).__init__()
        net=[ConvLayer(3,64,3,1,1,2,2,0)]
        net+=[ConvLayer(64,96,3,1,1,2,2,0)]
        net+=[ConvBlock(96,128,3,1,1)]
        net+=[ConvLayer(128,128,3,1,1,2,2,0)]
        net+=[ConvBlock(128,256,3,1,1)]
        net+=[ConvBlock(256,256,3,1,1)]
        self.convnet=nn.Sequential(*net)
        net=[nn.Dropout(p=0.5)]
        net+=[nn.Linear(256*12*12,2000)]
        net+=[nn.ReLU()]
        net+=[nn.Dropout(p=0.5)]
        net+=[nn.Linear(2000,7)]
        net+=[nn.Dropout(p=0.5)]
        self.fc=nn.Sequential(*net)
    def forward(self,x):
        y=self.convnet(x)
        y=y.flatten(start_dim=1) #[bsz,256*12*12]
        y=self.fc(y)
        return y