# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-11 22:21
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import math

from fastai.layers import *
from fastai.core import *
###############################################################################################
# Standard resnet

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=[3,3], downsample=None):
        super().__init__()

        if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes,kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x    # (128,128,1000)

        out = self.conv1(x) # (128,128,1000)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # (128,128,1000)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Sequential):
    '''1d adaptation of the torchvision resnet'''

    def __init__(self, block, layers, kernel_size=3, num_classes=2, input_channels=3, inplanes=64, fix_feature_dim=True,
                 kernel_size_stem=None, stride_stem=2, pooling_stem=True, stride=2, lin_ftrs_head=None, ps_head=0.5,
                 bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, use_ecgNet_Diagnosis="ecgNet"):
        self.inplanes = inplanes

        layers_tmp = []

        if (kernel_size_stem is None):
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size, list) else kernel_size
        # stem
        layers_tmp.append(nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem,
                                    padding=(kernel_size_stem - 1) // 2, bias=False))
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if (pooling_stem is True):
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        # backbone
        for i, l in enumerate(layers):
            if (i == 0):
                layers_tmp.append(self._make_layer(block, inplanes, layers[0], kernel_size=kernel_size))
            else:
                layers_tmp.append(
                    self._make_layer(block, inplanes if fix_feature_dim else (2 ** i) * inplanes, layers[i],
                                     stride=stride, kernel_size=kernel_size))

        # head
        # layers_tmp.append(nn.AdaptiveAvgPool1d(1))
        # layers_tmp.append(Flatten())
        # layers_tmp.append(nn.Linear((inplanes if fix_feature_dim else (2**len(layers)*inplanes)) * block.expansion, num_classes))
        # head有下面这些：
        # Sequential(
        #   (0): AdaptiveConcatPool1d(
        #     (ap): AdaptiveAvgPool1d(output_size=1)
        #     (mp): AdaptiveMaxPool1d(output_size=1)
        #   )
        #   (1): Flatten()
        #   (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (3): Dropout(p=0.25, inplace=False)
        #   (4): Linear(in_features=256, out_features=128, bias=True)
        #   (5): ReLU(inplace=True)
        #   (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (7): Dropout(p=0.5, inplace=False)
        #   (8): Linear(in_features=128, out_features=5, bias=True)
        # )
        if use_ecgNet_Diagnosis =="ecgNet":
            head = create_head1d((inplanes if fix_feature_dim else (2 ** len(layers) * inplanes)) * block.expansion,
                                 nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head,
                                 act=act_head, concat_pooling=concat_pooling)
            layers_tmp.append(head)

        super().__init__(*layers_tmp)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_layer_groups(self):
        return (self[6], self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self, x):
        self[-1][-1] = x

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)


def resnet1d_wang(**kwargs):
    if (not ("kernel_size" in kwargs.keys())):
        kwargs["kernel_size"] = [5, 3]
    if (not ("kernel_size_stem" in kwargs.keys())):
        kwargs["kernel_size_stem"] = 7
    if (not ("stride_stem" in kwargs.keys())):
        kwargs["stride_stem"] = 1
    if (not ("pooling_stem" in kwargs.keys())):
        kwargs["pooling_stem"] = False
    if (not ("inplanes" in kwargs.keys())):
        kwargs["inplanes"] = 128

    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)

if __name__ == '__main__':
    model = resnet1d_wang(num_classes=5, input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[128], inplanes=768, use_ecgNet_Diagnosis="false")
    result = model(torch.rand((64,12,1000)))   # (128,128,250)
    print(result.shape)