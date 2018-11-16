#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-7 
# @Author  : Schnee
# @File    : train.py


# coding:utf8
from config import opt
import torch as t
#import numpy as np
from data.load_data import Eye
from model.multi_model import M_model
from model.losses import MultiLoss
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from torchnet import meter
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import argparse
import time
import copy
import sys


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def getNetwork(args):
    if (args.net_type == 'alexnet'):
        net = models.alexnet(pretrained=args.finetune)
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        if(args.depth == 11):
            net = models.vgg11(pretrained=args.finetune)
        elif(args.depth == 13):
            net = models.vgg13(pretrained=args.finetune)
        elif(args.depth == 16):
            net = models.vgg16(pretrained=args.finetune)
        elif(args.depth == 19):
            net = models.vgg19(pretrained=args.finetune)
        else:
            print('Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'squeezenet'):
        net = models.squeezenet1_0(pretrained=args.finetune)
        file_name = 'squeeze'
    elif (args.net_type == 'resnet'):
        if (args.depth == 18):
            net = models.resnet18(pretrained=args.finetune)
        elif (args.depth == 34):
            net = models.resnet34(pretrained=args.finetune)
        elif (args.depth == 50):
            net = models.resnet50(pretrained=args.finetune)
        elif (args.depth == 101):
            net = models.resnet101(pretrained=args.finetune)
        elif (args.depth == 152):
            net = models.vgg152(pretrained=args.finetune)
        else:
            print('Error : resnet should have depth of either [18,34,50,101,152]')
            sys.exit(1)
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('Error : Network should be either [alexnet / squeezenet / vggnet / resnet]')
        sys.exit(1)

    return net, file_name

'''
def changeNetwork(args,model_ft):
    if (args.resetClassifier):
        print('| Reset final classifier...')
        print('| Add features of size %d' % opt.feature_size[0])
        #num_ftrs = model_ft.fc.in_features
        if (args.net_type=='resnet'):   #最后一层全连接 属性：（fc）
            num_ftrs = model_ft.fc.in_features

            #for i, param in enumerate(model_ft.parameters()):
            #    param.requires_grad = False  # 冻结参数的更新
            #num_ftrs=model_ft.fc.in_features

            model_ft.add_module('fc', nn.Linear(num_ftrs, opt.feature_size[0]))
            model_ft.add_module('bnn', nn.BatchNorm1d(opt.feature_size[0]))
            model_ft.add_module('relun', nn.ReLU(inplace=True))

        elif(args.net_type in ['alexnet','vgg']):
            pass
    return model_ft

def branchNetwork(args,feature_model):
    #task1
    Net = b_Net(args,feature_model)

'''
def train():

    # step1: configure model
    model_ft, file_name = getNetwork(args)
    model= M_model(model_ft, args)
   # model = getattr(models, opt.model)()
    print("load model--------------------------")
    print( model)
   # if opt.load_model_path:
    #    model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data
    train_data =Eye(opt.train_data_root, train=True)  # 训练集
    val_data = Eye(opt.train_data_root, train=False)  # 验证集

    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)
    dataloader={'train':train_dataloader,'val':val_dataloader}
    num={'train':train_data.__len__(),'val':val_data.__len__()}
    # step3: criterion and optimizer目标函数和优化器

    criterion= MultiLoss()

    #criterion = t.nn.CrossEntropyLoss()

    lr = opt.lr
    optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=opt.weight_decay)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # step4:  统计指标：平滑处理之后的损失
   #loss_meter = meter.AverageValueMeter()  # AverageValueMeter能够计算所有数的平均值和标准差,，这里用来统计一个epoch中损失的平均
   # confusion_matrix = meter.ConfusionMeter(2)  # 混淆矩阵
   # previous_loss = 1e100
    print("---------in train_model-----------")
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_acc1 = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    best_acc4 = 0.0
    alpha=0
    # train
    for epoch in range(opt.max_epoch):
        print('Epoch {}/{}'.format(epoch+1, opt.max_epoch))
        print('-' * 18)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects1 = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            running_corrects4 = 0.0
            # tqdm 加一个进度条
            #print(len(dataloader[phase]))
            #print(len(train_data))
            print(phase)
            print(num[phase])
            for ii, (data, label) in tqdm(enumerate(dataloader[phase]), total=len(dataloader[phase])):
                # train model
                data = Variable(data)
                label = Variable(label)
                if alpha >0:
                    weight=np.random.beta(alpha,alpha,opt.batch_size)
                   # print(weight)
                    x_weight = weight.reshape(opt.batch_size, 1, 1, 1)
                    x_weight=torch.Tensor(x_weight)
                    y_weight = weight.reshape(opt.batch_size, 1)
                    y_weight = torch.Tensor(y_weight)
                    #print(y_weight)
                    index = np.random.permutation(opt.batch_size)
                    print(index)
                    x1, x2 = data, data[index-1]
                    data = x1 * x_weight + x2 * (1 - x_weight)
                    y1, y2 = label, label[index]
                    label= y1 * y_weight + y2 * (1 - y_weight)
                    label=(label+0.5).int().float()
           
                if opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()

                optimizer.zero_grad()  # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
                task1_pred, task2_pred, task3_pred, task4_pred=model(data)
                    
                #-------------------------------------------------------
                t1 = []
                t2 = []
                t3 = []
                t4 = []
                for i, s_label in enumerate(label):
                    t1.append([float(abs(s_label[0] - 1))])
                    t2.append([float(abs(s_label[1] - 1))])
                    t3.append([float(abs(s_label[2] - 1))])
                    t4.append([float(abs(s_label[3] - 1))])
                t1 = torch.Tensor(t1).long().squeeze().cuda()
                t2 = torch.Tensor(t2).long().squeeze().cuda()
                t3 = torch.Tensor(t3).long().squeeze().cuda()
                t4 = torch.Tensor(t4).long().squeeze().cuda()
     
                loss = criterion(task1_pred, t1,task2_pred, t2,task3_pred, t3,task4_pred, t4)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

                #统计误差
                _, l1 = torch.max(task1_pred, 1)
                _, l2 = torch.max(task2_pred, 1)
                _, l3 = torch.max(task3_pred, 1)
                _, l4 = torch.max(task4_pred, 1)
                running_corrects1 += float(torch.sum(l1 == t1))
                running_corrects2 += float(torch.sum(l2 == t2))
                running_corrects3 += float(torch.sum(l3 == t3))
                running_corrects4 += float(torch.sum(l4 == t4))
                #print(running_corrects1)
                #print(running_corrects2)
                #print(running_corrects3)
                #print(running_corrects4)
            # meters update and visualize计算验证集上的指标及可视化

            epoch_loss = running_loss / num[phase]
            epoch_acc1 = running_corrects1 / num[phase]
            
            epoch_acc2= running_corrects2 / num[phase]
            epoch_acc3 = running_corrects3 / num[phase]
            epoch_acc4 = running_corrects4 / num[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Acc: {:.4f}Acc: {:.4f}Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc1, epoch_acc2, epoch_acc3, epoch_acc4))

            # deep copy the model
            epoch_acc=0.25*(epoch_acc1+epoch_acc2+epoch_acc3+epoch_acc4)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            if phase == 'val' and epoch_acc1 > best_acc1:
                best_acc1 = epoch_acc1
            if phase == 'val' and epoch_acc2 > best_acc2:
                best_acc2 = epoch_acc2
            if phase == 'val' and epoch_acc3 > best_acc3:
                best_acc3 = epoch_acc3
            if phase == 'val' and epoch_acc4 > best_acc4:
                best_acc4 = epoch_acc4

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}  {:4f}  {:4f}  {:4f}   {:4f}'.format(best_acc,best_acc1, best_acc2,best_acc3,best_acc4))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--net_type', default='resnet', type=str, help='model')
    parser.add_argument('--depth', default=18, type=int, help='depth of model')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')#0.0005
    parser.add_argument('--finetune', '-f', action='store_false', help='Fine tune pretrained model')
   # parser.add_argument('--addlayer', '-a', action='store_true', help='Add additional layer in fine-tuning')
   # parser.add_argument('--resetClassifier', '-r', action='store_false', help='Reset classifier')
   # parser.add_argument('--testOnly', '-t', action='store_false', help='Test mode with the saved model')

   # parser.add_argument('--task1',default=1,type=int,help='task1 branch number')
   # parser.add_argument('--task2', default=1, type=int, help='task2 branch number')
   # parser.add_argument('--task3', default=1, type=int, help='task3 branch number')
   # parser.add_argument('--task4', default=1, type=int, help='task4 branch number')

    args = parser.parse_args()

    model_ft=train()
    torch.save(model_ft, './-net'+str(args.net_type) +'-lr'+str(args.lr) + '.pkl')
