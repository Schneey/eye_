#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-7 
# @Author  : Schnee
# @File    : load_data.py

# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch
class Eye(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        '''
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/xxxxx_1,0.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('_')[-2].split('/')[-1]))
        imgs_num = len(imgs)
        # shuffle imgs
        np.random.seed(100)
        # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，
        # 则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，
        # 此时每次生成的随机数因时间差异而不同。
        imgs = np.random.permutation(imgs)  # shuffle 覆盖  permutation 不覆盖
        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            #加载训练集
            self.imgs = imgs[:int(0.7 * imgs_num)]
            imgs_num = len(imgs)
            self.num = imgs_num
 
        else:
            #加载验证集
            self.imgs = imgs[int(0.7 * imgs_num):]
            imgs_num = len(imgs)
            self.num = imgs_num

      
        # 数据转换操作，测试验证和训练的数据转换有所区别
        
        # 测试集和验证集不用数据增强
        if self.test or not train:
            self.transforms=T.Compose([
                T.Resize(227),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            # 训练集需要数据增强
        else:
            self.transforms = T.Compose([
                T.Resize(227),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(),
                T.RandomRotation(180),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        train: data/train/xxxxx_1,0,0,0.jpg
        '''
        img_path = self.imgs[index]
        s=self.imgs[index].split('.')[-2].split('_')[-1]
        l= []        #1,0,0,0
        for i, n in enumerate(s):
            if i % 2 == 0:
                l.append(int(n))
        label=torch.Tensor(l)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
'''
if __name__=='__main__':
    root='/home/zhangxueying/桌面/cat_dog/train'
    c=Eye(root)
    print(c.__getitem__(0))
'''
