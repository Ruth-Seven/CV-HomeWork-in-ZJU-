# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, cost_time
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

@cost_time
def train(config, model, train_dl, dev_dl, test_dl):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    dev_loss = None
    dev_acc = None
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        train_iter = iter(train_dl)
        for i, (trains, labels) in enumerate(train_iter):
            trains, labels = trains.to(config.device), labels.to(config.device)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)  # + F.mse_loss()
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                if dev_dl:
                    dev_acc, dev_loss = evaluate(config, model, dev_dl)
                # optim
                if dev_dl != None:
                    if dev_loss < best_loss:
                        best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                else:
                    train_loss = loss.item()
                    if train_loss < best_loss:
                        train_best_loss = train_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''

                # print each batch reslut
                time_dif = get_time_dif(start_time)
                print_batch_result(total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve)
                # write result to tensorboard
                writer.add_scalar("loss/train", loss.item(), total_batch)
                if dev_loss:
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                if dev_acc:
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("loss持续不下降，停止训练！")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_dl)




# 在test集合上评估模型效果
def test(config, model, test_dl):

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # 获取模型结果
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_dl, test=True)
    # 打印
    print("测试结果：")
    print(test_report)
    msg = 'Test accuracy: {0:>6.2%},  Test loss: {1:>5.2}'
    print(msg.format(test_acc, test_loss))
    print("混淆矩阵为:")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

import torch.nn.modules
# 根据数据评价模型效果，计算acc和loss，在test状态下生成分类结果和confusion matrix
def evaluate(config, model, data_dl, test=False):
    # 去除 dropout , norm标准化影响
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 其计算结果不进入torch计算图中
    with torch.no_grad():
        data_iter = data_dl
        for texts, labels in data_iter:
            # 模型在数据进行loss运算
            texts, labels = texts.to(config.device), labels.to(config.device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            # if test:
            #     if config.set_L2:
            #         loss += L2_penalty(model)
            #     elif config.set_L1:
            #         loss += L1_penalty(model)

            loss_total += loss
            # 获取运算结果，并取最大值为预测值
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # 将运算值添加到list
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # compute metrics
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



# print intermidiatelly result
def print_batch_result(total_batch, loss, train_acc, dev_loss, dev_acc, time_dif, improve):
    if dev_loss:
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
        print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
    else:
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Time: {3} {4} '
        print(msg.format(total_batch, loss.item(), train_acc,time_dif, improve))