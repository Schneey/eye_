#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-7 
# @Author  : Schnee
# @File    : config.py


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-20
# @Author  : Schnee
# @File    : config.py


# coding:utf8
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    #model = 'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test/'  # 测试集存放路径
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 63  # batch size
    use_gpu = True # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 200
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数
    feature_size=[256,128,128,128,128]
   # class_num=5

    task1_w=0.2
    task2_w=0.3
    task3_w=0.3
    task4_w=0.2


opt= DefaultConfig()
#opt.parse = parse




