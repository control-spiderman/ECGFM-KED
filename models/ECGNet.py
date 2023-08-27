# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-09 12:40

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
import os
import re
from collections import Counter
# import seaborn as sns
import sys
from scipy import signal
from torch.autograd import Variable


def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride, padding=(size - 1) // 2, bias=False)

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se = nn.Sequential(
            nn.Linear(planes, planes // 4),
            nn.ReLU(),
            nn.Linear(planes // 4, planes),
            nn.Sigmoid())

        self.shortcut = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes))

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        # out = self.relu(out)
        b, c, _ = out.size()
        y = nn.AdaptiveAvgPool1d(1)(out)
        y = y.view(b, c)
        y = self.se(y).view(b, c, 1)
        y = out * y.expand_as(out)

        out = y + residual
        return out


def conv_2d(in_planes, out_planes, stride=(1, 1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, size), stride=stride, padding=(0, (size - 1) // 2),
                     bias=False)


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se = nn.Sequential(
            nn.Linear(planes, planes // 4),
            nn.ReLU(),
            nn.Linear(planes // 4, planes),
            nn.Sigmoid())

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        # out = self.relu(out)
        batch_size, channel, _, _ = out.size()
        y = nn.AdaptiveAvgPool2d(1)(out)
        y = y.view(y.shape[0], -1)
        y = self.se(y).view(batch_size, channel, 1, 1)
        y = out * y.expand_as(out)

        out1 = y + residual
        return out1

class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=(1,1), padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=5, use_ecgNet_Diagnosis="ecgNet"):
        super(ECGNet, self).__init__()
        sizes = [
            [5, 5, 5, 5, 3, 3],
            [7, 7, 7, 7, 3, 3],
            [9, 9, 9, 9, 9, 9],
        ]
        self.sizes = sizes
        layers = [
            [3, 3, 2, 2, 2, 2],
            [3, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ]
        self.use_ecgNet_Diagnosis = use_ecgNet_Diagnosis
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(12, 60), stride=(1, 2), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 20), stride=(1, 2), padding=(0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 20), stride=(1, 2), padding=(0, 0), dilation=(1, 2))
        dilation_sizes = [(1, 2), (1, 5), (1, 7), (1, 12)]
        # self.conv3 = nn.ModuleList([
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0)),
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), dilation=(1, 6)),
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), dilation=(1, 12)),
        #     nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0), dilation=(1, 18)),
        # ])
        self.aspp1 = _ASPPModule(32, 32, (1, 1), padding=(0, 0), dilation=1)
        self.aspp2 = _ASPPModule(32, 32, (1, 3), padding=(0, 6), dilation=(1, 6))
        self.aspp3 = _ASPPModule(32, 32, (1, 3), padding=(0, 12), dilation=(1, 12))
        self.aspp4 = _ASPPModule(32, 32, (1, 3), padding=(0, 18), dilation=(1, 18))

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(32, 32, (1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Conv2d(32*5, 32, (1,1), bias=False)
        self.dropout = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()

        for i, size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1, 1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1, 1), size=sizes[i][1]))

            self.layers2.add_module('layer{}_2_1'.format(size),
                                    self._make_layer1d(BasicBlock1d, 64, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size),
                                    self._make_layer1d(BasicBlock1d, 64, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size),
                                    self._make_layer1d(BasicBlock1d, 128, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        # self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.sigmoid = nn.Sigmoid()
        self.w_omega = nn.Parameter(torch.Tensor(512, 512))
        self.u_omega = nn.Parameter(torch.Tensor(512, 1))

        self.leakyrelu = nn.LeakyReLU()
        self.feature_layer = nn.Sequential(
            nn.Linear(30, 30),
            nn.ReLU())

    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1, 2), size=3, res=True):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=(1, 1), padding=(0, 0), stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def attention_net(self, x):
        # [batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # [batch, seq_len, 1]
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        # [batch, seq_len, hidden_dim*2]
        scored_x = x * att_score
        # [batch, hidden_dim*2]
        context = torch.sum(scored_x, dim=1)
        return context

    def forward(self, x0):
        x0 = x0.unsqueeze(1)    # (8,12,1,5000)
        x0 = self.conv1(x0)     # # (8,32,1,2471)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)   # (8,32,1,1235)
        x0 = self.conv2(x0) # (8,32,1,608)
        x0 = self.bn2(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)   # (8,32,1,303)
        # x_1 = self.aspp1(x0)
        # x_2 = self.aspp2(x0)
        # x_3 = self.aspp3(x0)
        # x_4 = self.aspp4(x0)
        # x_5 = F.interpolate(self.avg_pool(x0), size=(x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)
        #
        # x0 = self.conv3(torch.cat((x_1, x_2, x_3, x_4, x_5), dim=1))
        # x0 = self.bn1(x0)
        # x0 = self.dropout(self.relu(x0))    # (8,32,1,303)

        pool, output = [], []
        for i in range(len(self.sizes)):
            # print(self.layers1_list[i])
            x = self.layers1_list[i](x0)    # (8,32,1,303)
            x = torch.flatten(x, start_dim=1, end_dim=2)    # (8,32,303)
            x = self.layers2_list[i](x) # (8,256,19)
            x = x.permute(0, 2, 1)  # (8,4,256)
            lstm_output, (final_hidden_state, final_cell_state) = self.lstm1(x) # (8,19,256)
            # attn_output = self.attention_net(lstm_output)
            lstm_output = lstm_output.permute(0, 2, 1)   # (8,256,19)
            # lstm_output = self.leakyrelu(lstm_output)

            pool_output = self.avgpool(lstm_output)  # (8,256,1)
            pool.append(pool_output)
            output.append(lstm_output)
        # out = torch.cat(xs, dim=2)
        pool_out = torch.cat(pool, dim=1)  #(8,768,1)
        pool_out = pool_out.view(pool_out.size(0), -1)
        out = torch.cat(output, dim=1)
        # features = self.feature_layer(features)
        # out = torch.cat([out, features], dim=1)
        if self.use_ecgNet_Diagnosis == "ecgNet":
            out = self.fc(pool_out)
            out = self.sigmoid(out)
        return out, pool_out  # (8,768)

if __name__ == '__main__':
    ecg_model = ECGNet(input_channel=1,use_ecgNet_Diagnosis="")
    signal = torch.rand((8, 12, 1000))
    ecg_model(signal)
