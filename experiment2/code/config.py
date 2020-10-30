

import json
import os
from pathlib import Path
import torch
import logging
from logging import Logger



logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Config(object):

    def __init__(self, dataset_path='./data'):
        print(f"\n_____________________________________\n{Config.__class__} confinging ")
        #dataset
        self.data_path =  Path(dataset_path)
        self.train_path = self.data_path / '/train.txt'  # 训练集
        self.dev_path = self.data_path / '/dev.txt'  # 验证集
        self.test_path = self.data_path  / '/test.txt'  # 测试集
        self.class_list = None
        self.num_classes = None

        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        # self.num_classes = len(self.class_list)  # 类别数

        self.dataset_workers = 0 # windwo 只能支持数据载入的线程为main threath

        #pre_train model
        # self.word_vectors_path = dataset + '/data/word_vectors.pkl'  # 预训练词向量

        # main model
        self.model_name = 'model'

        # you can change the default device for different computation devices.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'New using computation device {self.device}')

        self.log_path = './log/' + self.model_name  # log保存地址
        self.save_path = self.log_path + '/saved_dict/' + self.model_name + '.ckpt'  # 模型保存地址
        # Hyperparameters
        # self.filter_sizes = (5, 4, 3, 2)  # 每层的卷积核尺寸
        # self.num_filters = 64  # 卷积核数量(channels数)

        # train

        self.dropout = 0.5  # 随机失活
        self.weight_decay = 0.01  # 设置weight_decay
        self.num_epochs = 20  # epoch数
        self.batch_size = 512  # mini-batch大小
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.learning_rate = 1e-3  # 学习率
        self.shuffle = True     # shuffle in a epoch


    # 新添加的特性
    def def_visdom(self):
        import visdom
        self.cur_batch_win_opts = {
            'title': 'Epoch Loss Trace',
            'xlabel': 'Batch Number',
            'ylabel': 'Loss',
            'width': 1200,
            'height': 600,
        }

    def __str__(self):
        return self.model_name