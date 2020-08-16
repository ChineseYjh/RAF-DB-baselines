from networks.layers import *

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()
        self.vgg19=torchvision.models.vgg19()
        self.vgg19.classifier[6]=nn.Linear(4096,7)
    def forward(self,x):
        feats=self.vgg19.features(x)
        feats=self.vgg19.avgpool(feats)
        feats=torch.flatten(feats,1)
        y=self.vgg19.classifier(feats)
        return [y,feats]