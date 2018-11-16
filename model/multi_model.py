#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-9 
# @Author  : Schnee
# @File    : multi_model.py

from torch import nn
import torch.nn.functional as F

class M_model(nn.Module):
    """
    definition of CRNet
    """

    def __init__(self,model_ft,args):
        super(M_model, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.model = model_ft

        self.task1 = Net(model_ft,layer=1,num_cls=2)
        self.task2 = Net(model_ft,layer=1, num_cls=2)
        self.task3 = Net(model_ft,layer=1 ,num_cls=2)
        self.task4 = Net(model_ft, layer=1,num_cls=2)

    def forward(self, x):
        for name, module in self.model.named_children():
            if name != 'fc':
                x = module(x)

        task1_out = self.task1.forward(x.view(-1, self.num_flat_features(x)))
        task2_out = self.task2.forward(x.view(-1, self.num_flat_features(x)))
        task3_out = self.task3.forward(x.view(-1, self.num_flat_features(x)))
        task4_out = self.task4.forward(x.view(-1, self.num_flat_features(x)))

        return task1_out,task2_out,task3_out,task4_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Net(nn.Module):

    def __init__(self, model, layer=1,num_cls=2):
        super(Net, self).__init__()

        num_ftrs = model.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_cls)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.relu(self.fc2(x1))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x3 = self.fc3(x2)

        return x3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

