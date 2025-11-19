#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ER.py
@Contact :   ouyangqun5525@jxnu.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/12 21:31   ouyangqun    pain.nopain         None
'''

import torch.nn as nn
import torch.nn.functional as F
from BAM import BAM
from coordatt import CoordAtt
import os
import torch

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/data-output')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 最终的代码
class ER(nn.Module):
    def __init__(self):
        super(ER, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)

        # self.c1 = CoordAtt(32,32)
        self.BAM1 = BAM(32)
        self.max1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)

        # self.c2 = CoordAtt(128,128)
        self.BAM2 = BAM(128)
        self.max2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.bn6 = nn.BatchNorm2d(256)

        # self.c3 = CoordAtt(256,256)
        self.BAM3 = BAM(256)
        self.max3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.bn8 = nn.BatchNorm2d(512)

        # self.c4 = CoordAtt(512,512)
        self.BAM4 = BAM(512)
        self.max4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        # self.relu1 = torch.nn.ReLU()
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        # self.fc1 = nn.Linear(pain * 100 * 100, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # dropout训练

        # self.relu2 = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)  # dropout训练

        # self.relu3 = torch.nn.ReLU()
        # self.fc3 = nn.Linear(64, 7)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.BAM1(x)
        x = self.max1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.BAM2(x)
        x = self.max2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.BAM3(x)
        x = self.max3(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        # x = self.dropout1(x)

        x = self.BAM4(x)
        x = self.max4(x)
        # x = self.relu2(x)

        # x = x.contiguous().view(x.size()[0], -1)
        x = self.flatten(x)
        # x = self.relu1(x)
        x = self.fc1(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout1(x)

        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x



def demo():
    net  = ER()
    input = torch.randn(128,3,100,100)
    ouput  = net(input)
    print(ouput.shape)

if __name__ == '__main__':
    demo()



