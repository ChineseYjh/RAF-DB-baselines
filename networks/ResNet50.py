from networks.layers import *

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        self.net=torchvision.models.resnet50()
        self.features=nn.Sequential(*(list(self.net.children())[:-1]))
        self.fc=nn.Linear(2048,7)
    def forward(self,x):
        feats=self.features(x)
        feats=torch.flatten(feats,1)
        y=self.fc(feats)
        return [y,feats]