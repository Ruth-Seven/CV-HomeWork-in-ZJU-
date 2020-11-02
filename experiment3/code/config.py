

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

    def __init__(self, dir_path='../', model="model"):
        print(f"\n_____________________________________\n{Config.__class__} confinging ")
        #dataset
        self.dir_path = Path(dir_path)
        self.data_path =  self.dir_path / 'data'
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
        self.model_name = model

        # you can change the default device for different computation devices.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'New using computation device {self.device}')

        self.log_path = self.dir_path / 'log' / self.model_name  # log保存地址
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.log_path / 'saved_dict' / (self.model_name + '.ckpt')  # 模型保存地址
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_pic_path = self.save_path.parent / "pics"
        self.save_pic_path.mkdir(parents=True, exist_ok=True)
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


    def __str__(self):
        return self.model_name