from platform import architecture
from re import X
from turtle import right
import torch
import torch.nn as nn

from models.Tool import ChannelAttention, SpatialAttention
from models.Tool import ConvNormAct, EdgeEnhance

class MEFEBlock(nn.Module):
    def __init__(self,in_channels,padding,dilation,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=ConvNormAct(in_channels,in_channels//2,3,padding=padding,dilation=dilation)
        self.integrate=ConvNormAct(in_channels//2,in_channels//2,1)
        self.ee=EdgeEnhance(in_channels//2)

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        x=self.integrate(x)
        x=self.ee(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x
    
class MFF(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.fusion=ConvNormAct(in_channels*4,in_channels,3,padding=1)
        self.drop=nn.Dropout2d(0.2)
        self.local=ConvNormAct(in_channels,in_channels,1)
        self.ee=EdgeEnhance(in_channels)
        self.ca=ChannelAttention(in_channels)

    def forward(self,x):
        x=self.fusion(x)
        x=self.drop(x)
        x=self.local(x)
        x=self.ee(x)
        x=self.ca(x)
        return x

class MEFE(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.cfe1=MEFEBlock(in_channels,1,1)
        self.cfe2=MEFEBlock(in_channels,2,2)
        self.cfe3=MEFEBlock(in_channels,3,3)
        self.cfe4=MEFEBlock(in_channels,4,4)
        self.cfe5=MEFEBlock(in_channels,1,1,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.cfe6=MEFEBlock(in_channels,3,3,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.cfe7=MEFEBlock(in_channels,1,1,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.cfe8=MEFEBlock(in_channels,3,3,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        

    def forward(self,x):
        x1=self.cfe1(x)
        x2=self.cfe2(x)
        x3=self.cfe3(x)
        x4=self.cfe4(x)
        x5=self.cfe5(x)
        x6=self.cfe6(x)
        x7=self.cfe7(x)
        x8=self.cfe8(x)
        out=torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        return out

class MEED(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.mefe=MEFE(in_channels)
        self.mff=MFF(in_channels)
        self.relu=nn.ReLU()

    def forward(self,x):
        short_cut=x
        x=self.mefe(x)
        x=self.relu(self.mff(x)+short_cut)
        return x
    
class CFA(nn.Module):
    def __init__(self,in_channels1,in_channels2):
        super().__init__()
        self.pro=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvNormAct(in_channels1,in_channels2,1),
            ChannelAttention(in_channels2),
        )
        self.sig=nn.Sigmoid()

    def forward(self,x1,x2):
        x=self.pro(x1)*self.sig(x2)+x2
        return x



class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.num_layer=len(in_channels)
        self.meed=nn.ModuleList()
        for i_layer in range(self.num_layer):
            self.meed.append(MEED(in_channels[i_layer]))
        self.cfa=nn.ModuleList()
        for i_layer in range(self.num_layer-1):
            self.cfa.append(CFA(in_channels[i_layer+1],in_channels[i_layer]))
        
    def forward(self,x):
        x_list=[]
        input=x[-1]
        for i in range(-1, -len(self.msc)-1, -1):
            x_d=self.meed[i](input)
            x_list.append(x_d)
            if i!=-self.num_layer:
                input=self.cfa[i](x_d,x[i-1])
        return x_list



class PH(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Sequential(
            ConvNormAct(in_channels,1,1,norm=None,act=None),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x

class Final_Supervise(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.fs=nn.ModuleList()
        self.num_layer=len(in_channels)
        for i_layer in range(self.num_layer):
            self.fs.append(PH(in_channels[i_layer],scale_factor[i_layer]))

    def forward(self,x):
        x=x[::-1]
        x_list=[]
        for i in range(self.num_layer):
            x_list.append(self.fs[i](x[i]))
        return x_list
