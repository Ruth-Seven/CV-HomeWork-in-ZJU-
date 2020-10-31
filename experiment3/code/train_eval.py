# coding: UTF-8
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

from sklearn import metrics
import time
from utils import get_time_dif, cost_time
from tensorboardX import SummaryWriter
import torch.nn.modules
import torchsnooper
class Trainer(object):
    def __init__(self, config, model, cost_function=F.cross_entropy):
        self.config = config
        self.model = model
        self.cost_function = cost_function

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def evalue(self):
        raise NotImplementedError


class ClassiferTrainer(Trainer):


    @cost_time
    def train(self, train_dl, dev_dl, test_dl):
        start_time = time.time()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        total_batch = 0  # 记录进行到多少batch
        best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升

        dev_loss = None
        dev_acc = None
        writer = SummaryWriter(log_dir=str(self.config.log_path) + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1,self.config.num_epochs))
            # scheduler.step() # 学习率衰减
            train_iter = iter(train_dl)
            for i, (trains, targets) in enumerate(train_iter):
                trains, targets = trains.to(self.config.device), targets.to(self.config.device)

                outputs = self.model(trains)
                self.model.zero_grad()
                loss = self.cost_function(outputs, targets)  # + F.mse_loss()
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = targets.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    if dev_dl:
                        dev_acc, dev_loss = self.evaluate(self.config, self.model, dev_dl)
                    # optim
                    if dev_dl != None:
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            torch.save(self.model.state_dict(), str(self.config.save_path))
                            improve = '*'
                            last_improve = total_batch
                        else:
                            improve = ''
                    else:
                        train_loss = loss.item()
                        if train_loss < best_loss:
                            best_loss = train_loss
                            torch.save(self.model.state_dict(), str(self.config.save_path))
                            improve = '*'
                            last_improve = total_batch
                        else:
                            improve = ''

                    # print each batch reslut
                    time_dif = get_time_dif(start_time)
                    self.print_batch_result(total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve)
                    # write result to tensorboard
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    if dev_loss:
                        writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    if dev_acc:
                        writer.add_scalar("acc/dev", dev_acc, total_batch)
                    self.model.train()

                total_batch += 1
                if total_batch - last_improve > self.config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("loss持续不下降，停止训练！")
                    flag = True
                    break
            if flag:
                break
        writer.close()
        self.test(test_dl)

    # 在test集合上评估模型效果
    def test(self, test_dl):

        self.model.load_state_dict(torch.load(str(self.config.save_path)))
        self.model.eval()
        start_time = time.time()
        # 获取模型结果
        test_acc, test_loss, test_report, test_confusion = self.evaluate(test_dl, test=True)
        # 打印
        print("测试结果：")
        print(test_report)
        msg = 'Test accuracy: {0:>6.2%},  Test loss: {1:>5.2}'
        print(msg.format(test_acc, test_loss))
        print("混淆矩阵为:")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

    # 根据数据评价模型效果，计算acc和loss，在test状态下生成分类结果和confusion matrix
    def evaluate(self, data_dl, test=False):
        # 去除 dropout , norm标准化影响
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        targets_all = np.array([], dtype=int)
        # 其计算结果不进入torch计算图中
        with torch.no_grad():
            data_iter = data_dl
            for texts, targets in data_iter:
                # 模型在数据进行loss运算
                texts, targets = texts.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(texts)
                loss = self.cost_function(outputs, targets)
                # if test:
                #     if config.set_L2:
                #         loss += L2_penalty(model)
                #     elif config.set_L1:
                #         loss += L1_penalty(model)

                loss_total += loss
                # 获取运算结果，并取最大值为预测值
                targets = targets.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                # 将运算值添加到list
                targets_all = np.append(targets_all, targets)
                predict_all = np.append(predict_all, predic)

        # compute metrics
        acc = metrics.accuracy_score(targets_all, predict_all)
        if test:
            report = metrics.classification_report(targets_all, predict_all, target_names=self.config.class_list, digits=4)
            confusion = metrics.confusion_matrix(targets_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)


    # print intermidiatelly result
    def print_batch_result(self, total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve):
        if dev_loss:
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
            print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
        else:
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Time: {3} {4} '
            print(msg.format(total_batch, loss.item(), train_acc, time_dif, improve))



class SegmentationTrainer(Trainer):


    @cost_time
    # @torchsnooper.snoop()
    def train(self, train_dl, dev_dl, test_dl):
        start_time = time.time()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)

        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        total_batch = 0  # 记录进行到多少batch
        best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升

        dev_loss = None
        dev_acc = None
        writer = SummaryWriter(log_dir=str(self.config.log_path) + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            # scheduler.step() # 学习率衰减
            train_iter = iter(train_dl)
            for i, (trains, targets) in enumerate(train_iter):
                optimizer.zero_grad()
                trains, targets = trains.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(trains)
                # 0 - 1化
                # outputs = F.softmax(outputs, dim=1)

                loss = self.cost_function(outputs, targets)  # + F.mse_loss()
                loss.backward()
                optimizer.step()

                if total_batch % 100 == 0:
                    train_acc, iou = self._cal_batch_acc_ios(targets, outputs)
                    # Show the singal of improvement
                    improve = ''
                    if dev_dl:
                        dev_acc, dev_loss = self.evaluate(dev_dl)
                    # find and save the best model
                    if dev_dl != None:
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            torch.save(self.model.state_dict(), str(self.config.save_path))
                            improve = '*'
                            last_improve = total_batch
                    else:
                        train_loss = loss.item()
                        if train_loss < best_loss:
                            best_loss = train_loss
                            torch.save(self.model.state_dict(), str(self.config.save_path))
                            improve = '*'
                            last_improve = total_batch

                    # print each batch resluts
                    time_dif = get_time_dif(start_time)
                    self.print_batch_result(total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve)

                    # write results to tensorboard
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    if dev_loss:
                        writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    if dev_acc:
                        writer.add_scalar("acc/dev", dev_acc, total_batch)
                    self.model.train()


                total_batch += 1
                if total_batch - last_improve > self.config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("loss持续不下降，停止训练！")
                    flag = True
                    break
            if flag:
                break
        writer.close()
        self.test(test_dl)

    # 在test集合上评估模型效果
    def test(self, test_dl):

        self.model.load_state_dict(torch.load(str(self.config.save_path)))
        self.model.eval()
        # 获取模型结果
        accs, iou, loss= self.evaluate(test_dl)
        self._print_evalue_result(accs, iou, loss, mode="Test")


    # 根据数据评价模型效果，计算acc和loss，在test状态下生成分类结果和confusion matrix
    def evaluate(self, data_dl):
        # 去除 dropout , norm标准化影响
        self.model.eval()
        loss_total = 0
        target_list = []
        predict_list = []
        # 其计算结果不进入torch计算图中
        with torch.no_grad():
            data_iter = data_dl

            for datas, targets in data_iter:
                # 模型在数据进行loss运算
                datas, targets = datas.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(datas)
                loss = self.cost_function(outputs, targets)
                loss_total += loss

                # 获取运算结果，并取最大值为预测值
                # 显存不够，只能使用CPU
                predict_list.append(outputs.to("cpu"))
                target_list.append(targets.to("cpu"))


        accs, ious = self._cal_batch_acc_ios(torch.cat((target_list), 0), torch.cat((predict_list), 0))
        return accs, ious, loss_total / len(data_iter)

    def _tensor2numpy(self, data):
        if type(data) == list or type(data) == tuple:
            return np.array([ x.cpu().numpy() for x in data ])
        elif type(data) == np.ndarray:
            return data.cpu().numpy()
        elif type(data) == torch.Tensor:
            return data
        else:
            raise TypeError

    def _cal_batch_acc_ios(self, targets, outputs):
        """
        @input: Tensor or numpy or list of tenor or list
        """
        # targets = self._tensor2numpy(targets)
        # outputs = self._tensor2numpy(outputs)
        #
        targets = targets.to('cpu')
        outputs = outputs.to('cpu')
        N = targets.shape[0]
        predicts = outputs.permute(0, 2, 3, 1).reshape(-1, self.config.num_classes).argmax(axis=1).reshape(N, self.config.weight, self.config.height)
        targets = targets.permute(0, 2, 3, 1).reshape(-1, self.config.num_classes).argmax(axis=1).reshape(N, self.config.weight, self.config.height)
        acc = self.pixel_acc(targets, predicts)
        iou = self.iou(targets, predicts)
        # return mean_acc, mean_ious
        return acc, iou

  # borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
    # Calculates class intersections over unions
    def iou(self, pred, target):
        ious = []
        for cls in range(self.config.num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = pred_inds[target_inds].sum()
            union = pred_inds.sum() + target_inds.sum() - intersection
            if union == 0:
                ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
            else:
                ious.append((intersection * 1.0 / max(union, 1)))
            # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
        return ious

    def pixel_acc(self, pred, target):
        correct = (pred == target).sum()
        total = (target == target).sum()
        return (correct * 1.0 / total)



    def _print_evalue_result(self, acc, iou, loss, mode = ""):
        # 打印
        print("测试结果：")
        msg = '{0:s} accuracy: {1:>6.2%},  {0:s} loss: {2:>5.2}'
        print(msg.format(mode, acc, loss))
        print(mode, " IoU:\n")
        for i, item in enumerate(iou):
            print("{0:>5} {1:^6.4}".format(i, item))

        # print intermidiatelly result

    def print_batch_result(self, total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve):
        if dev_loss:
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
            print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
        else:
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Time: {3} {4} '
            print(msg.format(total_batch, loss.item(), train_acc, time_dif, improve))

