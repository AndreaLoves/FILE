#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ER.py
@Contact :   ouyangqun5525@jxnu.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/12 21:31   ouyangqun    pain.nopain         None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.BAM import BAM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 最终的代码
class ER(nn.Module):
    def __init__(self):
        super(ER, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = torch.nn.PReLU()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu2 = torch.nn.PReLU()
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.BAM1 = BAM(32)
        self.max1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 128, 3)
        self.relu3 = torch.nn.PReLU()
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.relu4 = torch.nn.PReLU()
        self.bn4 = torch.nn.BatchNorm2d(128)


        self.BAM2 = BAM(128)
        self.max2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128 ,256, 3)
        self.relu5 = torch.nn.PReLU()
        self.bn5 = torch.nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3)
        self.relu6 = torch.nn.PReLU()
        self.bn6 = torch.nn.BatchNorm2d(256)


        self.BAM3 = BAM(256)
        self.max3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.relu7 = torch.nn.PReLU()
        self.bn7 = torch.nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3)
        self.relu8 = torch.nn.PReLU()
        self.bn8 = torch.nn.BatchNorm2d(512)


        self.BAM4 = BAM(512)
        self.max4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # dropout训练
        self.relu2 = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = torch.nn.PReLU()
        self.dropout2 = nn.Dropout(p=0.5)  # dropout训练
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x= self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.BAM1(x)
        x = self.max1(x)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.BAM2(x)
        x = self.max2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.BAM3(x)
        x = self.max3(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.bn8(x)
        x = self.BAM4(x)
        x = self.max4(x)

        x = x.contiguous().view(x.size()[0], -1)


        x = self.fc1(x)
        x= self.dropout1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x

def test_BAM():
    model = ER()
    # print(model)
    x = torch.randn(256,1,100,100)
    x = model(x)
    print(x.shape)

if __name__ == '__main__':
    test_BAM()





