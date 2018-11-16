#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-9 
# @Author  : Schnee
# @File    : losses.py

import torch.nn as nn

class MultiLoss(nn.Module):
    """
    CRLoss definition
    """
    def __init__(self, task1_w=0.2, task2_w=0.3,task3_w=0.3, task4_w=0.2):

        super(MultiLoss, self).__init__()

        self.task1_w = task1_w
        self.task2_w = task2_w
        self.task3_w = task3_w
        self.task4_w = task4_w
        self.task_criterion = nn.CrossEntropyLoss()



    def forward(self, task1_pred, task1_class,task2_pred, task2_class,task3_pred, task3_class,task4_pred, task4_class):
        task1_loss = self.task_criterion(task1_pred, task1_class)
        task2_loss = self.task_criterion(task2_pred, task2_class)
        task3_loss = self.task_criterion(task3_pred, task3_class)
        task4_loss = self.task_criterion(task4_pred, task4_class)


        cr_loss = self.task1_w * task1_loss + \
                  self.task2_w * task2_loss + \
                  self.task3_w * task3_loss + \
                  self.task4_w * task4_loss
        return cr_loss
