import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init


# 对应的操作就是特征提取的操作
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
                # print("x0",x[0].shape)
                # print("x1",x[1].shape)
                extent = (x[1].size()[2])//(x[0].size()[2])
                if extent == 1:
                    pre_spatial_att = F.avg_pool2d(x[1],kernel_size=extent,stride=extent+1)
                elif(extent ==2 ):
                    pre_spatial_att = F.avg_pool2d(x[1],kernel_size=extent*3,stride=extent)
                elif(extent>2):
                    pre_spatial_att = F.avg_pool2d(x[1], kernel_size=extent, stride=extent)

            else:
                pre_spatial_att = x[1]


            x_compress = self.bnrelu(self.p1*self.compress(x[0])+self.p2*self.compress(pre_spatial_att))

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


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride)
        self.max1 = nn.MaxPool2d(2, 2)
        self.cbam = CBAM(in_planes, planes, 16)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x[0])))
        out = F.relu(self.bn1(self.conv2(out)))
        out = self.max1(out)
        out = self.cbam({0: out, 1: x[1], 2: x[2]})
        out0 = self.relu(out[0])

        return {0: out0, 1: out[1],2: out[2]}


class CSACNN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(CSACNN, self).__init__()
        self.in_planes = 64

        # 这里是输入层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 中间的网络主体部分，循环体
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=3)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=3)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=3)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=3)

        # 最后的输出层
        self.flatten = nn.Flatten()
        self.relu1 = torch.nn.ReLU()
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        # self.fc1 = nn.Linear(pain * 100 * 100, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # dropout训练
        self.relu2 = torch.nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)  # dropout训练
        self.relu3 = torch.nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = {0: out, 1: None, 2: None}

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out[0]
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fc1(out)
        # out = self.relu2(out)
        out = self.flatten(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu3(out)
        out = self.dropout1(out)

        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out


def network():
    return CSACNN(Block, [1,1,1,1])


def demo():
    net = network()
    # print(net)
    input = torch.randn(128, 3, 100, 100)
    ouput = net(input)
    # print(ouput.shape)

if __name__ == '__main__':
    demo()