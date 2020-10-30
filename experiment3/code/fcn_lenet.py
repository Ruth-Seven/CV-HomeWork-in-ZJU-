
from torch import nn
from config import Config
from leNet5 import *


from dataprocess import TransAndCacheDataset
from Mytransformer import AddBgandRes

class FCNConfig(Config):
    """配置参数"""
    def __init__(self,path="../"):
        super().__init__(path)
        self.batch_size = 64
        self.num_epochs = 20
        # 添加路径
        self.background_path = self.data_path /  "background"
        self.trans_data_path = self.data_path / "trans"
        self.trans_train_path = self.trans_data_path / "train"
        self.trans_test_path = self.trans_data_path / "test"

        self.weight = 28
        self.height = 28
        self.num_classes = 10

        #



class FCN_Lene32t(nn.Module):
    def __init__(self, config, LeNet):
        super().__init__()
        self.to(LeNetConfig.device)
        self.lenet = LeNet

        self.le_layer1 = self.lenet.CP1
        self.le_layer2 = self.lenet.CP2


        self.transconv1 = nn.ConvTranspose2d(120, 60, kernel_size=3, stride=3, padding=1, dilation=1) # 7 * 7

        self.transconv2 = nn.ConvTranspose2d(60, 30, kernel_size=3, stride=3, padding=4, dilation=1) # 14 * 14

        self.transconv3 = nn.ConvTranspose2d(30, 10, kernel_size=3, stride=3, padding=8, dilation=1 )  # 14 * 14

        self.classifier = nn.Conv2d(10, config.num_classes, kernel_size=1)


    def forward(self, x):
        x = self.transconv1(self.lenet(x))
        skip_link1 = x + self.le_layer2(x)
        x = self.transconv2(skip_link1)
        skip_link2 = x + self.le_layer1(x)
        x = self.transconv3(skip_link2)
        x = self.classifier(x)
        return x


if '__main__' == __name__:


    le_config = FCNConfig('../')
    # 下载源数据
    # PIL image 对象需要 transform
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    minst_train_dataset = torchvision.datasets.MNIST(str(le_config.data_path), download=True, train=True,
                                                     transform=None)
    minst_test_dataset = torchvision.datasets.MNIST(str(le_config.data_path), download=True, train=False,
                                                    transform=None)
    # 数据转化
    train_dataset = TransAndCacheDataset(le_config, minst_train_dataset, AddBgandRes(le_config), train=True, reload=True, transformer=transform)
    test_dataset = TransAndCacheDataset(le_config, minst_test_dataset, AddBgandRes(le_config), train=False, reload=True, transformer=transform)
    # 设置Dataload
    train_dl = dataloader.DataLoader(train_dataset, config.batch_size, shuffle=config.shuffle, num_workers=config.dataset_workers)
    test_dl = dataloader.DataLoader(test_dataset, config.batch_size, shuffle = config.shuffle, num_workers = config.dataset_workers)
    print(f"The length fo train dataset:{len(minst_train_dataset)}")
    print(f"The length fo test dataset:{len(minst_test_dataset)}")

    #建立模型
    lenet = LeNet(le_config)
    model = FCN_Lene32t(config, lenet)

    # 训练模型
    model = LeNet(le_config)
    cost_function = nn.BCEWithLogitsLoss()
    train(config, model, cost_function, train_dl, None, test_dl )



