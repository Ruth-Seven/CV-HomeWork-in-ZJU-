from torchvision import transforms
from numpy import random
from PIL import Image
import numpy as np
import torch
class AddBg(object):

    def __init__(self, config):
        self.config = config
        weight = config.weight
        height = config.height
        #         "../data/background"
        bg_path = config.background_path
        #         print(bg_path.exists())
        pics_path = []
        for pic in bg_path.iterdir():
            pics_path.append(pic)
        # pics_path
        ## 载入PIL image 并转化为RBG（R,G,B,3）图像
        imgs = [Image.open(str(path)).convert('RGB') for path in pics_path]
        self.imgs = [img for img in imgs if img.size[0] > weight and img.size[1] > height]
        self.size = len(imgs)
        # print(self.size)
        self.weight = weight
        self.height = height

    # def __call__(self, pic, label=None):
    #     raise NotImplementedError
    def __call__(self, pic, label):
        """
        argumenmt:
        @pic: Image instance(w,h)

        @return: Train Pic:new image instance(w,h,3) ,Target Pic a tensor Instance (w, h, 10)

        The size of pic should be (H, W)
        The function will return new image instance with random picture background.
        The background picture was place in the class config.background_path.
        Well, Let's begin!
        Ps. You should add a pipeline to solve the question that convert Image to tensor.
        """
        return self.get_trian_pic(pic), self.get_target_pic(pic, label, self.config.num_classes)

    def get_trian_pic(self, pic):
        pic.convert('RGB')
        bg_idx = np.random.randint(0, self.size)
        #         self.imgs[bg_idx].show()
        # 注意这要加self
        bg = self._getRectImage(self.imgs[bg_idx], self.weight, self.height)
        new_pic = self._gray2rgb(pic)
        return self._merge(new_pic, bg)

    def get_target_pic(self, pic, label, num_classes):
        raise NotImplementedError

    def _getRectImage(self, img, weight, height):
        # watch out the differenct between PIL.image.shape and np.array.shape
        W, H = img.size  # shape: (W, W)
        arr = np.array(img)  # shape: (H, W)

        rand_w = random.randint(0, W - weight - 1)
        rand_h = random.randint(0, H - height - 1)
        narr = arr[rand_h:rand_h + height, rand_w:rand_w + weight:]
        #         print(arr.shape)
        #         print(W, " ", H)
        #         print(rand_w, ' ', rand_h, ' ', weight + rand_w)
        #         print(narr.shape)
        #         print(arr)
        return Image.fromarray(narr)

    # img and bg should have the same size
    #         from functools import reduce
    def _merge(self, img, bg):
        arri = np.array(img)
        arrb = np.array(bg)
        # PIL shepe is just the size of pexel, not number.
        # print(arri.shape)
        # print(arrb.shape)
        # print("@@@@@@@@@")

        #     shape = arri.shape
        #         print(shape)
        #     sum_pixel = reduce(lambda x, y: x * y, shape, 1)
        #     arri = arri.resize(sum_pixel)
        #     arrb = arrb.resize(sum_pixel)
        #     arri.resize(shape)

        return Image.fromarray(np.where(arri != 0, arri, arrb))

    def _gray2rgb(self, img):
        """
        argurment:
        @img: PIL image instance:(weight, height, 1)
        @return: PIL imagePIL image instance:(weight, height, 3)
        """

        arr = np.array(img)
        # add a new dim
        newshape = list(arr.shape).append(3)
        #         arr[:, np.newaxis]
        arr2 = np.copy(arr)
        arr3 = np.copy(arr)
        newarr = np.dstack((arr, arr2, arr3))
        # print(newarr.shape)
        return Image.fromarray(newarr)



#
class AddBgandMulLabel(AddBg):



    def get_target_pic(self, pic, label, num_classes): # Image (28, 28) -> Narray(28, 28, 11)
        return self._get_multarget_pic(pic, label, num_classes)

    def _get_multarget_pic(self, pic, label, num_classes): # Image (28, 28) -> Narray(28, 28, 11)
        pic_arr = np.array(pic)
        new_shape = list(pic_arr.shape)
        new_shape.append(num_classes)
        arr = np.zeros(new_shape, dtype=np.float32)
        label_arr = np.where(pic_arr[:, :]>0, 1, 0)
        background_arr = np.where(pic_arr[:, :] > 0, 0, 1)
        arr[:,:, label] = label_arr
        arr[:, :, num_classes - 1] = background_arr
        return torch.tensor(arr, dtype=torch.float32)


class AddBgandBFlabel(AddBg):

    def get_target_pic(self, pic, label, num_classes=2): # Image (28, 28) -> Narray(28, 28, 2)
        return self._get_bftarget_pic(pic, num_classes)

    def _get_bftarget_pic(self, pic, num_classes=2): # Image (28, 28) -> Narray(28, 28, 11)
        """
        return the target pic that only catch the difference between background pic and foreground pic.
        """
        pic_arr = np.array(pic)
        new_shape = list(pic_arr.shape)
        new_shape.append(num_classes)
        arr = np.zeros(new_shape, dtype=np.float32)
        label_arr = np.where(pic_arr>0, 1, 0)
        background_arr = np.where(pic_arr> 0, 0, 1)
        arr[:, :, 0] = label_arr
        arr[:, :, 1] = background_arr
        # Image.fromarray(arr.astype(np.uint8)).show()
        # arr
        # ar3[:, :, 0:2] = arr * 255
        # Image.fromarray(ar3.astype(np.uint8)).show()
        # tmep = np.zeros((28,28,3))
        # tmep[:,:,0:2] = target
        # Image.fromarray(tmep.astype(np.uint8)).show()
        return torch.tensor(arr, dtype=torch.float32)
