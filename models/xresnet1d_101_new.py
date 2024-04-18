# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2024-02-11 20:57

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

"""
Implementation based on pytorch, It not used in this research.
"""


def conv_layer(ni, nf, ks=3, stride=1, padding=None, bn=True, act=nn.ReLU):
    """Create a convolutional layer with optional batch normalization and activation."""
    if padding is None:
        padding = (ks - 1) // 2
    layers = [nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm1d(nf))
    if act:
        layers.append(act())
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    """Basic residual block with size adjustment in the shortcut."""
    def __init__(self, ni, nf, stride=1, ks=3, act=nn.ReLU):
        super(ResBlock, self).__init__()
        self.conv1 = conv_layer(ni, nf, ks, stride=stride, act=act)
        self.conv2 = conv_layer(nf, nf, ks, act=None)

        # Shortcut connection with size adjustment
        if ni == nf and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(ni, nf, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(nf)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + self.shortcut(x))

class XResNet1d(nn.Module):
    """XResNet1d model."""
    def __init__(self, layers, input_channels=12, num_classes=1000, kernel_size=5):
        super(XResNet1d, self).__init__()
        nf = 64
        # Initial convolution layers
        self.starter = nn.Sequential(
            conv_layer(input_channels, nf // 2, ks=kernel_size, stride=2),  # 步幅从2调整为1
            conv_layer(nf // 2, nf // 2, ks=kernel_size),
            conv_layer(nf // 2, nf, ks=kernel_size)
        )

        # stride_nums = [2,2,2,2]
        stride_nums = [2,2,1,1]
        # ResBlocks
        block_szs = [64, 128, 256, 512, 768]
        self.num_features = block_szs[-1]
        # block_szs = [64, 64, 128, 256, 512]
        self.blocks = nn.Sequential(
            # *[self._make_layer(nf*(2**i), nf*(2**(i+1)), blocks=l, stride=stride_nums[i])
            *[self._make_layer(block_szs[i], block_szs[i+1], blocks=l, stride=stride_nums[i])
              for i, l in enumerate(layers)]
        )

        # Adaptive concatenation pool and a final linear layer
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(block_szs[-1], num_classes)
        )
        self.layers = layers

    def get_num_layer(self, name=""):
        if name.startswith("starter"):
            # 初始卷积层被视为一层
            return 1
        elif name.startswith("blocks"):
            # "blocks.x.y" -> 第x个ResBlock（从0开始计数）, 每个ResBlock只计为一层
            block_id = int(name.split('.')[1])
            # 对于每个ResBlock，返回其ID加上初始卷积层，以得到正确的层数
            return block_id + 1
        else:
            # 对于模型头部和其他任何不在"starter"或"blocks"路径下的组件，
            # 它们放在所有ResBlock之后，因此返回总ResBlock数量加上初始层
            total_blocks = len(self.layers)  # self.layers定义了每个阶段的ResBlock数量
            return total_blocks + 1

    def _make_layer(self, ni, nf, blocks, stride):
        return nn.Sequential(*[ResBlock(ni if i == 0 else nf, nf, stride=stride if i == 0 else 1) for i in range(blocks)])

    def forward(self, x):
        # x = x.reshape(x.shape[0], 1, -1)
        x = self.starter(x) #(64,12,4096) -> (64,64,4094); (64,1,60000)->(64,64,30000)
        x = self.blocks(x)  #(64,64,2048) -> (64,1024,512)
        # x = self.head(x)
        return x

def xresnet1d101(input_channels=12, num_classes=1000, kernel_size=5):
    """Constructs a ResNet-101 model."""
    return XResNet1d([3, 4, 23, 3], input_channels, num_classes, kernel_size)

def count_parameters(model):
    """统计模型的参数量"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def create_resnet_model():
    ecg_model = xresnet1d101(num_classes=2, input_channels=12, kernel_size=5)
    return ecg_model

if __name__ == '__main__':
    model = xresnet1d101(num_classes=2, input_channels=12, kernel_size=5)
    total_params = count_parameters(model)  # 42,496,384  13944069
    print(f"Total parameters: {total_params}")
    result = model(torch.rand(64, 12, 5000))
    print(result.shape)