from models.BAM import BAM
from models import *
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):#__init()中必须自己定义可学习的参数
        super(ResidualBlock, self).__init__()  #调用nn.Module的构造函数

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, stride),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, stride),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        self.max1 = nn.MaxPool2d(2, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max1(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        # self.inchannel = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.max1 = nn.MaxPool2d(2, 2)
        self.BAM1 = BAM(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.max2 = nn.MaxPool2d(2, 2)
        self.BAM2 = BAM(128)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.max3 = nn.MaxPool2d(2, 2)
        self.BAM3 = BAM(256)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.max4 = nn.MaxPool2d(2, 2)
        self.BAM4 = BAM(512)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.dropout1 = nn.Dropout(p=0.5)  # dropout训练
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)  # dropout训练
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, channels, stride):
        layers = []
        layers.append(block(self.inchannel, channels, stride))
        self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.BAM1(out)
        out = self.max1(out)

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.BAM2(out)
        out = self.max2(out)

        out = self.conv5(out)
        out = self.conv6(out)

        out = self.BAM3(out)
        out = self.max3(out)

        out = self.conv7(out)
        out = self.conv8(out)

        out = self.BAM4(out)
        out = self.max4(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout1(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out


# def ResNet18():
#     return ResNet(ResidualBlock,num_classes=2)

def demo():
    net = ResNet18()
    summary(net,(128,3,100,100))
    # print(net)
    # input = torch.randn(128, 3, 100, 100)
    # ouput = net(input)
    # print(ouput.shape)


if __name__ == '__main__':
    demo()