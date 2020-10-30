from config import *
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

from Mytransformer import *


class TransAndCacheDataset(Dataset):

    def __init__(self, config, dataset, pre_processer: AddBgandRes, train=True, reload=False,transformer=None):
        super().__init__()

        self.config = config
        self.data_path = config.trans_train_path if train else config.trans_test_path
        #
        self.reload = reload
        self._dataset = dataset
        self._dataLoader = None
        self.pre_processer = pre_processer
        self.transformer = transformer
        # init dataset
        self.num_classes = config.num_classes
        self.data = []
        self.create_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.transformer:
            return self.transformer(self.data[item])
        else:
            return self.data[item] #Image, tensor


    def create_dataset(self):
        if self._is_save() and not self.reload:
            return
        else:
            self._save_pic_from_dataset()
            self._load_dataaset()

    def _is_save(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        try:
            if len(self.data_path.iterdir()) == self.num_classes:
                return True
            else:
                return False
        except:
            return False
        return True


    def _save_pic_from_dataset(self):
        k = 0
        for img, label in self._dataset:
            data_img, label_ten = self.pre_processer(img, label)
            path = self.data_path / str(label) / str(k)
            path.mkdir(parents=True, exist_ok=True)
            data_img.save(path / "x.png" )
            # np.save(path / "y.txt", label_arr)
            torch.save(label_ten, path / "y")

            k += 1
            if k % 1000 == 0:
                print(f"已保存{k}张图片和Label")

    def _load_dataset(self):
        for label in range(self.num_classes):
            path = self.data_path / str(label)
            for dir_path in path.iterdir():
                x_img = Image.open(path / "x.png")
                # y_arr = np.load("y.txt")
                y_ten = torch.load(path / "y")
                self.data.append({"X":x_img, "Y":y_ten, "l":dir_path.name})


    #  sample = {'X': img, 'Y': target, 'l': label}

    # def get_dataset(self):
    #     self._set_dataset()
    #     self._set_data_loader()
    #     return self._dataLoader


    # def _set_dataset(self):
    #     ...
    #
    #
    # def _set_data_loader(self):
    #     ...
    #


if "__main__" == __name__:
    from leNet5 import LeNetConfig
    from fcn_lenet import FCNConfig
    config = LeNetConfig('../')
    minst_train_dataset = MNIST(str(config.data_path), download=True, train=True)

    dataset = TransAndCacheDataset(FCNConfig(), minst_train_dataset, AddBgandRes, reload=True)
    print(len(dataset))
    dataset[10]["X"].show()
    dataset[10]["Y"].show()
    print(dataset[10]["l"].show())