#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ER.py
@Contact :   ouyangqun5525@jxnu.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/12 21:31   ouyangqun    pain.nopain         None
'''

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

import os
import torch

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/data-output')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0 \
            , dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size \
                , stride=stride, padding=padding, dilation=dilation \
                , groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01 \
                , affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, gate_channels, reduction_ratio=8 \
            , pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        if in_channels != gate_channels:
            self.att_fc = nn.Sequential(
                nn.Conv2d(in_channels,gate_channels, kernel_size=1),
                nn.BatchNorm2d(gate_channels),
                nn.ReLU(inplace=True)
            )
        self.alpha = nn.Sequential(
            nn.Conv2d(2, 1,bias=False, kernel_size=1),
            nn.LayerNorm(gate_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = inputs[0]
        b, c, _, _ = x.size()
        pre_att = inputs[1]
        channel_att_sum = None
        if pre_att is not None:
            if hasattr(self, 'att_fc'):
                pre_att = self.att_fc(pre_att)
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                if pre_att is not None:
                    print("avg_pool", avg_pool.view(b, 1, 1, c).shape)
                    print("avg_poolpre", avg_pool.shape)
                    print("pre_att", self.avgpool(pre_att).size())
                    print("pre_att", self.avgpool(pre_att).view(b, 1, 1, c).size())
                    avg_pool = torch.cat((avg_pool.view(b, 1, 1, c), self.avgpool(pre_att).view(b, 1, 1, c)), dim=1)
                    avg_pool = self.alpha(avg_pool).view(b, c)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.maxpool(x)
                if pre_att is not None:
                    max_pool = torch.cat((max_pool.view(b, 1, 1, c), self.maxpool(pre_att).view(b, 1, 1, c)), dim=1)
                    max_pool = self.alpha(max_pool).view(b, c)
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2) \
                .unsqueeze(3).expand_as(x)

        out = x*scale
        # It can be only one, we did not optimize it due to lazy.
        # Will be optimized soon.
        return {0:out,1:out}

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1) \
                .unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,in_channel, gate_channels):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1 \
                , padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
        self.p1 = Parameter(torch.ones(1))
        self.p2 = Parameter(torch.zeros(1))
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x[1] is None:
            x_compress = self.compress(x[0])
        else:
            if x[1].size()[2]!=x[0].size()[2]:
                extent = (x[1].size()[2])//(x[0].size()[2])
                if extent == 1:
                    pre_spatial_att = F.avg_pool2d(x[1],kernel_size=extent,stride=extent+1)
                else:
                    pre_spatial_att = F.avg_pool2d(x[1],kernel_size=extent,stride=extent)

            else:
                pre_spatial_att = x[1]


            x_compress = self.bnrelu(self.p1*self.compress(x[0])+self.p2*self.compress(pre_spatial_att))
            # print("x_compress.size()", x_compress.size())
            # print("over!!")
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        # It can be only one, we did not optimize it due to lazy.
        return {0:x[0] * scale,1:x[0] * scale}

class CBAM(nn.Module):
    def __init__(self, in_channel,gate_channels, reduction_ratio=16 \
            , pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(in_channel,gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(in_channel,gate_channels)
    def forward(self, x):

        x_out = self.ChannelGate({0:x[0],1:x[1]})
        channel_att = x_out[1]
        if not self.no_spatial:
            x_out = self.SpatialGate({0:x_out[0],1:x[2]})
        return {0:x_out[0],1:channel_att,2:x_out[1]}




class ER(nn.Module):
    def __init__(self):
        super(ER, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.CBAM1 = CBAM(32,128,16)
        self.max1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)

        self.CBAM2 = CBAM(128,256,16)
        self.max2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.bn6 = nn.BatchNorm2d(256)

        self.CBAM3 = CBAM(256,512,16)
        self.max3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)

        self.CBAM4 = CBAM(512,512,16)
        self.max4 = nn.MaxPool2d(2, 2)

        self.relu1 = torch.nn.ReLU()
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # dropout训练
        self.relu2 = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)  # dropout训练
        self.relu3 = torch.nn.ReLU()
        # self.fc3 = nn.Linear(64, 7)
        self.fc3 = nn.Linear(64, 2)



    def forward(self, x):
        x = {0: x, 1: None, 2: None}
        out = self.conv1(x[0])
        out = self.bn1(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # out = self.avg1(out)
        x[0] = out
        x = self.CBAM1(x)
        out = self.max1(x[0])


        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu2(out)
        out = self.conv4(out)

        out = self.bn4(out)
        out = self.relu2(out)

        out = self.avg1(out)
        x[0] = out
        x = self.CBAM2(x)
        out = self.max2(x[0])


        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu2(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu2(out)

        out = self.avg1(out)
        x[0] = out
        x = self.CBAM3(x)
        out = self.max3(x[0])


        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu2(out)

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu2(out)

        out = self.avg1(out)
        x[0] = out
        x = self.CBAM4(x)
        out = self.max4(x[0])
        # x = self.relu2(x)

        out = out.contiguous().view(out.size()[0], -1)

        # x = self.relu1(x)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        out = self.relu3(out)
        out = self.fc3(out)
        out = F.log_softmax(out,dim=1)

        return out



def demo():
    net  = ER()
    input = torch.randn(128,3,100,100)
    ouput  = net(input)
    print(ouput.shape)

    # x1 = torch.randn(128, 64, 1, 1)
    # b, c, _, _ = x1.size()
    # net = nn.AdaptiveAvgPool2d(1)
    # x2 = torch.randn(128, 64, 25, 25)
    # x2 = net(x2)
    # x1 = net(x1)
    # print(x2.shape)
    #
    # temp1 = torch.cat((x1.view(b, 1, 1, c) , x2.view(b, 1, 1, c)), dim=1)
    # print(temp1.size())

if __name__ == '__main__':
    demo()



