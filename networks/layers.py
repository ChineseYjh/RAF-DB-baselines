import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,input_size,output_size,kernel,stride,padding):
        super(ConvBlock,self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.kernel=kernel
        block=[nn.Conv2d(input_size,output_size,kernel,stride,padding)]
        block+=[nn.ReLU()]
        self.block=nn.Sequential(*block)
    def forward(self,x):
        return self.block(x)

class ConvLayer(nn.Module):
    def __init__(self,input_size,output_size,kernel,stride,padding,pool_kernel,pool_stride,pool_padding):
        super(ConvLayer,self).__init__()
        self.pool_kernel=pool_kernel
        self.pool_stride=pool_stride
        self.pool_padding=pool_padding
        layer=[ConvBlock(input_size,output_size,kernel,stride,padding)]
        layer+=[nn.MaxPool2d(pool_kernel,pool_stride,pool_padding)]
        self.layer=nn.Sequential(*layer)
    def forward(self,x):
        return self.layer(x)