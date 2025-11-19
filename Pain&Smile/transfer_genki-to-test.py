#%% 导入模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
import time
import copy
from models.model import ER
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/ouyangqun/data-output')


#%% 数据预处理及增强
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(100),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(100),
        # transforms.CenterCrop(100),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


#%% 制作数据集
image_datasets = {
    x: datasets.ImageFolder(
        root=os.path.join('/home/ouyangqun/emo_rec/emo_rec/dataset/ck/newck', x),
        transform=data_transforms[x]
    ) for x in ['train', 'val']
}
#%% 制作数据加载器
dataloaders = {
    x: DataLoader(
        dataset=image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=0
    ) for x in ['train', 'val']
}
#%% 数据集大小查看，类名查看，选择训练设备
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
'''
{'train': 244, 'val': 153}
['ants', 'bees']
device(type='cuda', index=0)
'''
#%% 训练数据可视化
inputs, labels = next(iter(dataloaders['train']))
grid_images = torchvision.utils.make_grid(inputs)




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
    # plt.savefig('./Confusion_ck.png')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.show()



def no_normalize(im):
    im = im.permute(1, 2, 0)
    # im = im*torch.Tensor([0.229, 0.224, 0.225])+torch.Tensor([0.485, 0.456, 0.406])
    return im

grid_images = no_normalize(grid_images)
plt.title([class_names[x] for x in labels])
plt.imshow(grid_images)
plt.show()
# %% 训练模型函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    t1 = time.time()
    pre_val = []
    x_val = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        print(
            f'EPOCH: {epoch+1:0>{len(str(num_epochs))}}/{num_epochs}',
            f'LR: {lr:.4f}',
            end=' '
        )
        # 每轮都需要训练和评估
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()   # 将模型设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度归零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs.argmax(1)
                    loss = criterion(outputs, labels)

                    # 反向传播+参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'train':
                        truelabel = outputs.argmax(axis=-1)

                        for i in truelabel.cpu():
                            pre_val.append(i.item())
                        for j in labels.cpu():
                            x_val.append(j.item())
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum()
            if phase == 'train':
                # 调整学习率
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 打印训练过程
            if phase == 'train':
                print(
                    f'LOSS: {epoch_loss:.4f}',
                    f'ACC: {epoch_acc:.4f} ',
                    end=' '
                )
                writer.add_scalar('train_acc', epoch_acc, epoch)
                writer.add_scalar('train_loss', epoch_loss, epoch)
            else:
                print(
                    f'VAL-LOSS: {epoch_loss:.4f}',
                    f'VAL-ACC: {epoch_acc:.4f} ',
                    end='\n'
                )
                writer.add_scalar('test_acc', epoch_acc, epoch)
                writer.add_scalar('test_loss', epoch_loss, epoch)
            # 深度拷贝模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


            if phase == 'train' and epoch_acc > best_acc:
                plot_confuse(pre_val, x_val)
                print("x_val",x_val)
                print("pre_val",pre_val)


    t2 = time.time()
    total_time = t2-t1
    print('-'*10)
    print(
        f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s',
        f'BEST-VAL-ACC: {best_acc:.4f}'
    )
    # 加载最佳的模型权重
    model.load_state_dict(best_model_wts)
    return model
#%% 测试结果可视化函数
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloaders['val']))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(1)

        plt.figure(figsize=(9, 9))
        for i in range(inputs.size(0)):
            plt.subplot(2,2,i+1)
            plt.axis('off')
            plt.title(f'pred: {class_names[preds[i]]}|true: {class_names[labels[i]]}')
            im = no_normalize(inputs[i].cpu())
            plt.imshow(im)
        plt.savefig('train.jpg')
        plt.show()
# #%% 训练模型：参数微调
# # 加载预训练模型

# model_ft = models.resnet18(pretrained=False)


#这里是刚开始的训练的参数
# model_ft = models.resnet18(pretrained=False)
# # pretrained_dict = torch.load("/home/ouyangqun/emo_rec/emo_rec/dataset/model.pt")
# pretrained_dict = torch.load("/home/ouyangqun/emo_rec/emo_rec/dataset/model.pt")
# # model_ft.load_state_dict(pretrained_dict)
#
# model_dict = model_ft.state_dict()
# # 重新制作预训练的权重，主要是减去参数不匹配的层，楼主这边层名为“fc”
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
# # 更新权重
# model_dict.update(pretrained_dict)
# model_ft.load_state_dict(model_dict)


# 这里是我们的模型的迁移的设置
model_ft = ER()
# # pretrained_dict = torch.load("/home/ouyangqun/emo_rec/emo_rec/dataset/model.pt")
model_ft.load_state_dict(torch.load('/home/ouyangqun/emo_rec/emo_rec/dataset/params_fer2013.pkl'))
#
#
# 获取resnet18的全连接层的输入特征数
num_ftrs = model_ft.fc3.in_features

# 调整全连接层的输出为2
model_ft.fc3 = nn.Linear(num_ftrs, len(class_names))

# 将模型放到GPU/CPU
model_ft = model_ft.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 选择优化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9,weight_decay=5e-4)

# 定义学习器调整策略，每5轮学习率下调0.1个因子
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.3)

# 调用训练函数训练
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=250
)

# 测试结果可视化
visualize_model(model_ft)
torch.save(model_ft.state_dict(), '/home/ouyangqun/emo_rec/emo_rec/dataset/all_train_params.pkl')
#%% 训练模型：特征提取
# 加载预训练模型
# model_conv =ER()
# model_conv.load_state_dict(torch.load('/home/ouyangqun/emo_rec/emo_rec/dataset/params_gebki0308.pkl'))
# model_conv = models.resnet18(pretrained=True)
#
# # 冻结除全连接层外的所有层, 使其梯度不会在反向传播中计算
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# # 获取resnet18的全连接层的输入特征数
# # num_ftrs = model_conv.fc3.in_features
# num_ftrs = model_conv.fc.in_features
#
# # 调整全连接层的输出特征数为2
# # model_conv.fc3 = nn.Linear(num_ftrs, 2)
# model_conv.fc = nn.Linear(num_ftrs, 2)
#
# # 将模型放到GPU/CPU
# model_conv = model_conv.to(device)
#
# # 定义损失函数
# criterion = nn.CrossEntropyLoss()
#
# # 选择优化器, 只传全连接层的参数
# # optimizer_conv = optim.SGD(model_conv.fc3.parameters(), lr=1e-3, momentum=0.9)
# optimizer_conv = optim.SGD(model_conv.parameters(), lr=1e-3, momentum=0.9)
# # optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.0001, betas=(0.9, 0.99))
#
# # 定义优化器器调整策略，每5轮后学习率下调0.1个乘法因子
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.5)
#
# # 调用训练函数训练
# model_conv = train_model(
#     model_conv,
#     criterion,
#     optimizer_conv,
#     exp_lr_scheduler,
#     num_epochs=100
# )
# # 测试结果可视化
# visualize_model(model_conv)
# %% 保存模型
# torch.save(model_conv.state_dict(), '/home/ouyangqun/emo_rec/emo_rec/dataset/model.pt')