
from torch import nn
from config import Config
from leNet5 import *


from dataprocess import TransAndCacheDataset
from Mytransformer import AddBgandMulLabel
from leNet5 import LeNetConfig
from train_eval import SegmentationTrainer
from utils import transform


#Config
class PreLeNetConfig_11c(LeNetConfig):
    def __init__(self, model="premodel11"):
        super().__init__(model=model)

class FCNConfig_11c(Config):
    """配置参数"""
    def __init__(self,path="../", model="fcnmodel_ch11", pre_path=None):
        super().__init__(path, model)
        self.batch_size = 64
        self.num_epochs = 12
        # 添加路径
        self.background_path = self.data_path /  "background"
        self.trans_data_path = self.data_path / "trans_ch11"
        self.trans_train_path = self.trans_data_path / "train"
        self.trans_test_path = self.trans_data_path / "test"

        self.pre_model_path = pre_path

        self.weight = 28
        self.height = 28
        self.num_classes = 11
        self.require_improvement = 3000

# Model
class Pre_Lenet_11c(LeNet):


    def __init__(self, lenetconfig):
        super().__init__(lenetconfig)

        self.CP1 = ConvandPool(lenetconfig.input_channels, 6, 5, 1, 2, 2, 2) # 14 * 14
        self.CP2 = ConvandPool(6, 16, 5, 1, 2, 2,2) # 7 * 7
        self.CP3 = ConvandPool(16, 120, 5, 1, 2, 2, 2)# 3 * 3

        self.fc1 = nn.Linear(3 * 3 * 120, 100)
        self.fc2 = nn.Linear(100, 10)

        # 设置 data tpye of model weight
        self.to(lenetconfig.device)


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

class FCN_Lene32t_11c(nn.Module):

    def __init__(self, config, lenet):
        super().__init__()
        #载入预训练模型
        self.lenet = lenet
        if hasattr(config, "pre_model_path"):
            print(f"载入模型{config.pre_model_path.name}")
            self.lenet.load_state_dict(torch.load(config.pre_model_path))

        self.transconv1 = nn.ConvTranspose2d(120, 16, kernel_size=3, stride=3, padding=1, dilation=1) # 7 * 7
        self.batch1 = nn.BatchNorm2d(16)
        self.transconv2 = nn.ConvTranspose2d(16, 6, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1) # 14 * 14
        self.batch2 = nn.BatchNorm2d(6)
        self.transconv3 = nn.ConvTranspose2d(6, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # 28 * 28
        self.batch3 = nn.BatchNorm2d(3)
        self.classifier = nn.Conv2d(3, config.num_classes, kernel_size=1)
        self.to(config.device)

    def forward(self, x):
        layer_out = self.lenet.forward(x)

        x = self.transconv1(layer_out["CP3"])
        x = self.batch1(x)

        skip_link2 = x + layer_out["CP2"]
        x = self.transconv2(skip_link2)
        x = self.batch2(x)

        skip_link1 = x + layer_out["CP1"]
        x = self.transconv3(skip_link1)
        x = self.batch3(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x


if '__main__' == __name__:
    pre_leconfig = PreLeNetConfig_11c()
    fcn_config = FCNConfig_11c(pre_path=pre_leconfig.save_path)
    # 下载源数据
    # PIL image 对象需要 transform

    minst_train_dataset = torchvision.datasets.MNIST(str(fcn_config.data_path), download=True, train=True,
                                                     transform=None)
    minst_test_dataset = torchvision.datasets.MNIST(str(fcn_config.data_path), download=True, train=False,
                                                    transform=None)
    # 数据转化
    reload = True
    train_dataset = TransAndCacheDataset(fcn_config, minst_train_dataset, AddBgandMulLabel(fcn_config), train=True,
                                         reload=reload, transformer=transform, target_transformer=None)
    test_dataset = TransAndCacheDataset(fcn_config, minst_test_dataset, AddBgandMulLabel(fcn_config), train=False,
                                        reload=reload, transformer=transform,  target_transformer=None)

    print("Data shape:", train_dataset[0][0].shape)
    print("Target shape:", train_dataset[0][1].shape)
    # 设置Dataload
    train_dl = dataloader.DataLoader(train_dataset, fcn_config.batch_size,
                                     shuffle=fcn_config.shuffle, num_workers=fcn_config.dataset_workers)
    test_dl = dataloader.DataLoader(test_dataset, fcn_config.batch_size,
                                    shuffle = fcn_config.shuffle, num_workers =fcn_config.dataset_workers)

    #建立模型

    ##if  Pre-trained model need train process
    # if not pre_leconfig.save_path.exists():
    #     err_info = f"没有已存在的预训练模型，请去leNet模型设置model名{pre_leconfig.model},或者移动一个对应的已经训练好的模型到目录{pre_leconfig.save_path}"
    #     raise AssertionError(err_info)

    lenet = Pre_Lenet_11c(pre_leconfig)
    model = FCN_Lene32t_11c(fcn_config, lenet)

    # 训练模型

    train = SegmentationTrainer(fcn_config, model, nn.BCEWithLogitsLoss(reduction="mean"))
    train.train(train_dl, None, test_dl )

    # #visual model
    # visual_model(fcn_config, model)

