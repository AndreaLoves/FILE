
from __future__ import print_function
import torch.optim as optim
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from data.CK import CK
from models import *
from models.model import ER
import time
# from resnet_cbam_dca import dca_cbam_resnet50
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/ouyangqun/data-output')

parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
parser.add_argument('--fold', default=4, type=int, help='k fold number')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch nopain or last checkpoint epoch

learning_rate_decay_start = 20  # 50
learning_rate_decay_every = 4 # 5
learning_rate_decay_rate = 0.8 # nopain.9

cut_size = 100
total_epoch = 100

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


trainset = CK(split = 'Training', fold = opt.fold, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
testset = CK(split = 'Testing', fold = opt.fold, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)

# Model
if opt.model == 'VGG19':
    net = ER()
    # net = network()
    # net = dca_cbam_resnet50()
    # net = models.resnet50()
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 7)
    # net = VGG('VGG19')
    # net = dca_cbam_resnet50()
elif opt.model == 'Resnet18':
    # net = ResNet18()
    print(11)

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'Test_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda(1)


# 显示混淆矩阵
def plot_confuse(predictions, y_val):
    #获得真实标签
    truelabel = y_val  # 将one-hot转化为label
    # truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    # 指定分类类别
    classes = range(np.max(truelabel)+1)
    classes=["anger" ,"disgust", "fear", "happy", "sad" ,"surprise" ,"contempt"]
    title='    '
   #混淆矩阵颜色风格
    cmap=plt.cm.jet
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
   # 按照行和列填写百分比数据
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/ouyangqun/emo_rec/emo_rec/dataset/Confusion_ck.jpg')
    plt.show()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
# optimizer =torch.optim.Adam(net.parameters(),lr=opt.lr)
# Training
def train(epoch,net):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(1), targets.cuda(1)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()
        writer.add_scalar('train_loss', train_loss, epoch)
        # writer.close()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    Train_acc = 100.*correct/total
    writer.add_scalar('Train_acc', Train_acc, epoch)
    # writer.close()
    print("Train_acc",Train_acc)

def test(epoch,net):
    x_val = []
    pre_val = []
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    classnum = 7
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(1), targets.cuda(1)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)

        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        for i in predicted.cpu():
            pre_val.append(i.item())
        for j in targets.cpu():
            x_val.append(j.item())


        utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # 下面是用来计算各种评价指标的
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)

    # Save checkpoint.
    Test_acc = 100.*correct/total
    writer.add_scalar('Test_acc', Test_acc, epoch)
    # writer.close()
    print("Test_acc",Test_acc)




    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = acc_num.sum(1) / target_num.sum(1)

    # 打印格式方便复制
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('F1', " ".join('%s' % id for id in F1))
    print('accuracy', accuracy)
    # for id in recall:
    #     writer.add_scalars('recall', id, epoch)
    # for id in precision:
    #     writer.add_scalars('precision', id, epoch)
    # for id in F1:
    #     writer.add_scalars('F1', id, epoch)
    # for id in accuracy:
    #     writer.add_scalars('accuracy', id, epoch)

    if Test_acc > best_Test_acc:
        # print('Saving..')
        # print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.dataset + '_' + opt.model):
            os.mkdir(opt.dataset + '_' + opt.model)
        if not os.path.isdir(path):
            os.mkdir(path)
        # torch.save(net.state_dict(), '/home/ouyangqun/emo_rec/emo_rec/dataset/params_.pkl')
        # torch.save(state, '/home/ouyangqun/emo_rec/emo_rec/dataset/Test_model.t7')
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch
        plot_confuse(pre_val, x_val)
        print("pre_val",pre_val)
        print("x_val",x_val)
t1 =time.time()
for epoch in range(start_epoch, total_epoch):
    train(epoch,net)
    test(epoch,net)

    t2 = time.time()
    total_time = t2 - t1
    print('-' * 10)
    print(
        f'TOTAL-TIME: {total_time // 60:.0f}m{total_time % 60:.0f}s',
    )
print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)
