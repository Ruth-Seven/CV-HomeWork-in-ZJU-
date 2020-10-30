

from torch import nn
from torch.nn.functional import softmax,relu
from config import *
from utils import transform
class LeNetConfig(Config):
    """配置参数"""
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.num_epochs = 20

class ConvandPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,\
                 stride=1, padding=2, pool_size=2, pool_striker=2 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size, pool_striker)

    def forward(self,x):
        x = self.conv(x)

        x = self.relu(x)
        x = self.pool(x)
        return x

class LeNet(nn.Module):
    def __init__(self, LeNetConfig):
        super().__init__()

        # reduce the number of fil
        self.CP1 = ConvandPool(1, 6, 5, 1, 2, 2, 2)
        self.CP2 = ConvandPool(6, 16, 5, 1, 2, 2,2)
        self.CP3 = ConvandPool(16, 120, 5, 1, 2, 2, 2)

        self.fc1 = nn.Linear(3 * 3 * 120, 100)
        self.fc2 = nn.Linear(100, 10)

        # 设置 data tpye of model weight
        self.to(LeNetConfig.device)


    def forward(self, x):
        x = self.CP1(x)
        x = self.CP2(x)
        x = self.CP3(x)

        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        # x = softmax(x, 1)
        return x


import torchvision
from torch.utils.data import dataset, dataloader
from train_eval import *
if '__main__' == __name__:

    config = LeNetConfig()
    minst_train_dataset = torchvision.datasets.MNIST('./data', download=True, train=True,transform=transform)
    minst_test_dataset = torchvision.datasets.MNIST('./data', download=True, train=False,transform=transform)
    # PIL image 对象需要 transform
    train_dl = dataloader.DataLoader(minst_train_dataset, config.batch_size, shuffle=config.shuffle, num_workers=config.dataset_workers)
    test_dl = dataloader.DataLoader(minst_test_dataset, config.batch_size, shuffle=config.shuffle, num_workers=config.dataset_workers)
    model = LeNet(config)

    print(f"The length fo train dataset:{len(minst_train_dataset)}")
    print(f"The length fo test dataset:{len(minst_test_dataset)}")

    train(config, model, train_dl, None, test_dl )



