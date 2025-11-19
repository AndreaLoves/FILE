# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#忽略警告
import warnings
warnings.filterwarnings('ignore')



# 打平操作
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# 通道部分的实现代码
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        # gate_channel输入通道数量, reduction_ratio = 16 , num_layers = 1叠加的层数
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        gate_channels = [gate_channel]  # eg 64
        gate_channels += [gate_channel // reduction_ratio] * num_layers  # eg 4
        gate_channels += [gate_channel]  # 64
        # gate_channels: [64, 4, 4]
        # 最后一层不需要激活函数，加上从0开始，所以是-2，而不是-pain
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                'gate_c_fc_%d' % i,
                nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
                                   nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        # 最后一个层的处理
        self.gate_c.add_module('gate_c_fc_final',
                               nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # avg_pool = f.avg_pool2d(x, x.size(2).item(), stride=x.size(2).item())
        avg_pool = f.avg_pool2d(x, x.size(2), stride=x.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


# 空间注意力机制代码
class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=16,
                 dilation_conv_num=2,
                 dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()

        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,
                      gate_channel // reduction_ratio,
                      kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0',
                               nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # 进行多个空洞卷积，丰富感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d' % i,
                nn.Conv2d(gate_channel // reduction_ratio,
                          gate_channel // reduction_ratio,
                          kernel_size=3,
                          padding=dilation_val,
                          dilation=dilation_val))
            self.gate_s.add_module(
                'gate_s_bn_di_%d' % i,
                nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        self.gate_s.add_module(
            'gate_s_conv_final',
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, x):
        return self.gate_s(x).expand_as(x)


# 融合代码
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self,).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, x):
        att = 1 + f.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x

def testaa():
    x = torch.randn(128,32,100,100)
    # print(x)
    model = BAM(32)
    out  = model(x)
    print(out.shape)

if __name__ == '__main__':
    testaa()