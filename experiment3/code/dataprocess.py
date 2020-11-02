from config import *
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from  tqdm import tqdm

from PIL import Image
from Mytransformer import *

from visualize import biggerpic_show
class TransAndCacheDataset(Dataset):
    def __init__(self, config, dataset, pre_processer: AddBg, train=True, reload=False,transformer=None,  target_transformer=None, justread=False):
        super().__init__()

        self.config = config
        self.data_path = config.trans_train_path if train else config.trans_test_path
        self.save_data =  self.data_path / ("train_data." if train else "test_data." + "ph")
        self.save_target = self.data_path / ("train_target." if train else "test_target." + "ph")
        self.mode = "train dataset" if train else "test dataset"
        #
        self.justread = justread
        self.reload = reload
        self._dataset = dataset
        self._dataLoader = None
        self.pre_processer = pre_processer
        self.transformer = transformer
        self.target_transformer = target_transformer
        # init dataset
        self.num_classes = config.num_classes
        self.data = [] # [ Img_1, Img_N]
        self.target = []  # [Label(W,H,10)......]
        self.len = 0

        # functional test
        self.test_model = False

        # preprocess
        self.create_dataset()

        # self.__get
        if train:
            self.sample_show()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        return self._transform(self.data[idx]), self.target[idx]

    def _transform(self, item):
        if self.transformer:
            return self.transformer(item)
        else:
            return item #Image, tensor

    def _target_tranformer(self, item):
        if self.target_transformer:
            return self.target_transformer(item)
        else:
            return item #Image, tensor

    def create_dataset(self):
        if self.justread:
            print("Just read, no read from files or save to files")
            self._save_pic_to_dataset()

        elif self._is_save() and not self.reload:
            print(self.mode,": 直接载入数据")
            self._load_dataset()
        else:
            print(self.mode, ": 转化数据，保存图片并载入数据")
            self._save_pic_to_dataset()

        self.len = len(self.data)


    def _is_save(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        if self.save_target.exists()  and self.save_data.exists():
            return True
        else:
            return False

    # new version
    def _save_pic_to_dataset(self):
        k = 0
        for img, label in tqdm(self._dataset):
            data_img, label_ten = self.pre_processer(img, label)
            self.data.append(data_img)

            self.target.append(label_ten.permute([2, 0, 1]))
            k += 1

            if self.test_model and k == 5 :
                break

        if not self.justread:
            self.data_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.data,  self.save_data)
            torch.save(self.target, self.save_target)
            print(f"已保存和载入{k}张图片和Label")
        else:
            print(f"已载入{k}张图片和Label")
    # new version
    def _load_dataset(self):
        self.data = torch.load(self.save_data)
        self.target = torch.load(self.save_target)
        assert np.array(self.data[0]).shape == (28, 28, 3)
        print(f"共载入{len(self.data)}幅图片")

    # old version
    #由于file descriptors的数量有限，无法读写大量打开的公共的文件夹，具体解决方法没有花很多时间去寻找，于是决定写到一个文件内。
    # def _save_pic_from_dataset(self):
    #     k = 0
    #     for img, label in self._dataset:
    #         data_img, label_ten = self.pre_processer(img, label)
    #         path = self.data_path / str(label)
    #         path.mkdir(parents=True, exist_ok=True)
    #         self.data.append({"X": data_img, "Y": label_ten, "l": label})
    #         k += 1
    #
    #     torch.save(self.data)
    #     print(f"已保存{k}张图片和Label")

    # old version
    # def _load_dataset(self):
    #     k = 0
    #     for label in range(self.num_classes):
    #         path = self.data_path / str(label)
    #         for pic_path in path.glob("*.png"):
    #             print(k)
    #             x_img = Image.open(pic_path)
    #             idx = pic_path.name.split("x")[0]
    #             # with open(str(path / f"{idx}y.pt"), "rb") as f:
    #             #     y_ten = pickle.load(f)
    #             # y_ten = torch.load(path / (str(idx) + "y.pt"))
    #             y_ten = np.load(path / f"{idx}y.pt")
    #             self.data.append({"X":x_img, "Y":y_ten, "l":pic_path.name})
    #             k += 1
    #     print(f"共载入{k}幅图片")

    #


    def sample_show(self):
        idx = [random.randint(0, self.len) for _ in range(100)]
        dataset = [x for x, _ in self._dataset]
        biggerpic_show(dataset, idx, self.data_path / "before.png")
        biggerpic_show(self.data, idx, self.data_path / "after.png")


if "__main__" == __name__:
    from leNet5 import LeNetConfig
    from fcn_lenet_11c import FCNConfig
    config = LeNetConfig('../')
    minst_train_dataset = MNIST(str(config.data_path), download=True, train=True)

    dataset = TransAndCacheDataset(FCNConfig(), minst_train_dataset, AddBgandRes, reload=True)
    print(len(dataset))
    dataset[10]["X"].show()
    dataset[10]["Y"].show()
    print(dataset[10]["l"].show())