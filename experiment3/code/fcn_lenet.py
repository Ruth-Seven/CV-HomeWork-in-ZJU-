
from torch import nn
from config import Config
from leNet5 import *


from dataprocess import TransAndCacheDataset
from Mytransformer import AddBgandRes
from leNet5 import LeNet
from train_eval import SegmentationTrainer
class FCNConfig(Config):
    """配置参数"""
    def __init__(self,path="../", model="fcnmodel", pre_path=""):
        super().__init__(path, model)
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
        self.pre_model_path = pre_path

class Pre_Lenet(LeNet):


    def forward(self, x):
        pre_x = super().forward(x)
        layer_out = {}
        x1 = self.CP1(x)
        x2 = self.CP2(x1)
        x3 = self.CP3(x2)
        layer_out["CP1"] = x1
        layer_out["CP2"] = x2
        layer_out["CP3"] = x3
        layer_out["y"] = pre_x
        return layer_out

class FCN_Lene32t(nn.Module):
    def forward(self, x):
        layer_out = self.lenet.forward(x)

        x = self.transconv1(layer_out["CP3"])

        skip_link2 = x + layer_out["CP2"]
        x = self.transconv2(skip_link2)

        skip_link1 = x + layer_out["CP1"]
        x = self.transconv3(skip_link1)
        x = self.classifier(x)
        return x

    def __init__(self, config, lenet):
        super().__init__()
        #载入预训练模型
        self.lenet = lenet
        if hasattr(config, "pre_model_path"):
            print(f"载入模型{config.pre_model_path.name}")
            self.lenet.load_state_dict(torch.load(config.pre_model_path))

        self.transconv1 = nn.ConvTranspose2d(120, 16, kernel_size=3, stride=3, padding=1, dilation=1) # 7 * 7

        self.transconv2 = nn.ConvTranspose2d(16, 6, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # 14 * 14

        self.transconv3 = nn.ConvTranspose2d(6, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # 28 * 28

        self.classifier = nn.Conv2d(3, config.num_classes, kernel_size=1)
        self.to(config.device)


if '__main__' == __name__:
    le_config = LeNetConfig()
    fcn_config = FCNConfig(pre_path=le_config.save_path)
    # 下载源数据
    # PIL image 对象需要 transform
    transform = transforms.Compose([

        transforms.ToTensor()
    ])
    minst_train_dataset = torchvision.datasets.MNIST(str(fcn_config.data_path), download=True, train=True,
                                                     transform=None)
    minst_test_dataset = torchvision.datasets.MNIST(str(fcn_config.data_path), download=True, train=False,
                                                    transform=None)
    # 数据转化
    reload = False
    train_dataset = TransAndCacheDataset(fcn_config, minst_train_dataset, AddBgandRes(fcn_config), train=True, reload=reload, transformer=transform, target_transformer=None)
    test_dataset = TransAndCacheDataset(fcn_config, minst_test_dataset, AddBgandRes(fcn_config), train=False, reload=reload, transformer=transform,  target_transformer=None)

    print("Data shape:", train_dataset[0][0].shape)
    print("Target shape:", train_dataset[0][1].shape)
    # 设置Dataload
    train_dl = dataloader.DataLoader(train_dataset, fcn_config.batch_size, shuffle=fcn_config.shuffle, num_workers=fcn_config.dataset_workers)
    test_dl = dataloader.DataLoader(test_dataset, fcn_config.batch_size, shuffle = fcn_config.shuffle, num_workers =fcn_config.dataset_workers)
    print(f"The length fo train dataset:{len(minst_train_dataset)}")
    print(f"The length fo test dataset:{len(minst_test_dataset)}")

    #建立模型

    lenet = Pre_Lenet(le_config)
    model = FCN_Lene32t(fcn_config, lenet)

    # 训练模型
    train = SegmentationTrainer(fcn_config, model, nn.BCEWithLogitsLoss())
    train.train(train_dl, None, test_dl )



