

from torch import nn
from torch.nn.functional import softmax,relu
from config import *
from utils import transform
class LeNetConfig(Config):
    """配置参数"""
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.num_epochs = 40

class LeNet(nn.Module):
    def __init__(self, LeNetConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(int(8*28*28), 28*28)
        self.fc1 = nn.Linear(7 * 7 * 8, 28 * 28)
        self.fc2 = nn.Linear(28*28, 10)

        # 设置 data tpye of model weight
        self.to(LeNetConfig.device)


    def forward(self, x):

        conv1 = self.conv1(x)
        conv1 = self.pool1(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.pool1(conv2)

        # flat the input from the dim 1
        # print(conv2.shape)
        flattedx = torch.flatten(conv2, 1)
        # print(flattedx.shape)
        # fc layers
        fc1 = self.fc1(flattedx)
        fc1 = relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = relu(fc2)
        return fc2



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



