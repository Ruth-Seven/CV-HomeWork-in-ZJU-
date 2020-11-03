
from train_eval import *

from pathlib import Path
from utils import transform
from  torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
import torch

from fcn_lenet_11c import FCNConfig_11c, FCN_Lene32t_11c, Pre_Lenet_11c, PreLeNetConfig_11c
from fcn_lenet_2c import FCNConfig_2c, FCN_Lene32t_2c, Pre_Lenet_2c, PreLeNetConfig_2c

from visualize import biggerpic_show
from dataprocess import TransAndCacheDataset
from Mytransformer import AddBgandBFlabel,AddBgandMulLabel
from torchvision.datasets import MNIST
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

def visual_model(fcn_config, model, test=False):

    #config
    PICS_NUM = 100
    #dataset
    minst_train_dataset = MNIST(str(fcn_config.data_path), download=True, train=False, transform=None)
    mnist_idx=[random.randint(0, len(minst_train_dataset)) for _ in range(PICS_NUM * 3)]

    subset = Subset(minst_train_dataset, mnist_idx)
    if fcn_config.num_classes == 2:
        train_dataset = TransAndCacheDataset(fcn_config, subset, AddBgandBFlabel(fcn_config),
                                            train=True,transformer=transform,justread=True)
    else:
        train_dataset = TransAndCacheDataset(fcn_config, subset, AddBgandMulLabel(fcn_config),
                                             train=True, transformer=transform, justread=True)
    train_dl = DataLoader(train_dataset, PICS_NUM, shuffle=False)


    #predict data
    model.load_state_dict(torch.load(fcn_config.save_path))

    if test:
        model_path = fcn_config.save_path.parent / (fcn_config.model_name + "forIoU.ckpt")
        model.load_state_dict(torch.load(fcn_config.save_path.parent / (fcn_config.model_name + "forIoU.ckpt")))
    else:
        model_path = fcn_config.save_path

    print(f"载入模型：{model_path}")

    model.eval()
    train_datas, target_data = next(iter(train_dl))
    predicts = model(train_datas.to(fcn_config.device)).cpu().detach()
    predicts_val = predicts.permute(0,2,3,1) * 255
    targets_val = target_data.permute(0,2,3,1) * 255
    assert (predicts_val > 255).sum() == 0 and (predicts_val < 0).sum() == 0
    assert (targets_val > 255).sum() == 0 and (targets_val < 0).sum() == 0



    origin_data = []
    for i in range(PICS_NUM):
        origin_data.append(subset[i][0])

    idx = list(range(PICS_NUM))
    biggerpic_show(origin_data, idx, fcn_config.save_pic_path / "TestData.png")
    biggerpic_show(targets_val, idx, fcn_config.save_pic_path / "TargetData.png")
    biggerpic_show(predicts_val, idx, fcn_config.save_pic_path / "TestResult.png")

if '__main__' == __name__:
    print("Visualization Test.")

    le_config = PreLeNetConfig_2c()
    fcn_config = FCNConfig_2c(pre_path=le_config.save_path)
    lenet = Pre_Lenet_2c(le_config)
    model = FCN_Lene32t_2c(fcn_config, lenet)
    visual_model(fcn_config, model)

    pre_leconfig = PreLeNetConfig_11c()
    fcn_config = FCNConfig_11c(pre_path=pre_leconfig.save_path)
    lenet = Pre_Lenet_11c(pre_leconfig)
    model = FCN_Lene32t_11c(fcn_config, lenet)
    visual_model(fcn_config, model)

    pre_leconfig = PreLeNetConfig_11c()
    fcn_config = FCNConfig_11c(pre_path=pre_leconfig.save_path)
    lenet = Pre_Lenet_11c(pre_leconfig)
    model = FCN_Lene32t_11c(fcn_config, lenet)
    visual_model(fcn_config, model, test=True)