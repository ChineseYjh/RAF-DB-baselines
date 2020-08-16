from networks.layers import *

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.net=torchvision.models.alexnet()
        self.net.classifier[6]=nn.Linear(4096,7)
    def forward(self,x):
        feats=self.net.features(x)
        feats=self.net.avgpool(feats)
        feats=torch.flatten(feats,1)
        y=self.net.classifier(feats)
        return [y,feats]