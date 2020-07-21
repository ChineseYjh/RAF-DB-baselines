from networks.layers import *

class DLPCNN(nn.Module):
    """
    return list[unnormalized tensor [bsz,7], deep feature tensor [bsz,2000]]
    """
    def __init__(self):
        super(DLPCNN,self).__init__()
        net=[ConvLayer(3,64,3,1,1,2,2,0)]
        net+=[ConvLayer(64,96,3,1,1,2,2,0)]
        net+=[ConvBlock(96,128,3,1,1)]
        net+=[ConvLayer(128,128,3,1,1,2,2,0)]
        net+=[ConvBlock(128,256,3,1,1)]
        net+=[ConvBlock(256,256,3,1,1)]
        self.convnet=nn.Sequential(*net)
        net=[nn.Linear(256*12*12,2000)]
        net+=[nn.ReLU()]
        net+=[nn.Linear(2000,7)]
        self.fc=nn.Sequential(*net)
    def forward(self,x):
        y1=self.convnet(x)
        y1=y1.flatten(start_dim=1) #[bsz,256*12*12]
        y2=self.fc(y1)
        return [y2,y1]