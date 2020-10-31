from torchvision import transforms
from numpy import random
from PIL import Image
import numpy as np
import torch
class AddBg(object):

    def __init__(self, config):
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

    def __call__(self, pic):
        """
        argumenmt:
        @pic: Image instance(w,h)

        @return: new Image instance(w,h,3)

        The size of pic should be (H, W)
        The function will return new image instance with random picture background.
        The background picture was place in the class config.background_path.
        Well, Let's begin!
        Ps. You should add a pipeline to solve the question that convert Image to tensor.
        """
        pic.convert('RGB')

        bg_idx = np.random.randint(0, self.size)
        #         self.imgs[bg_idx].show()
        #注意这要加self
        bg = self.getRectImage(self.imgs[bg_idx], self.weight, self.height)
        return self.merge(self.gray2rgb(pic), bg)

    def getRectImage(self, img, weight, height):
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
    def merge(self, img, bg):
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

    def gray2rgb(self, img):
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




class AddBgandRes(AddBg):

    def __init__(self, config):
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

    def __call__(self, pic, label):
        """
        argumenmt:
        @pic: Image instance(w,h)

        @return: new image instance(w,h,3) , a tensor Instance (w, h, 10)

        The size of pic should be (H, W)
        The function will return new image instance with random picture background.
        The background picture was place in the class config.background_path.
        Well, Let's begin!
        Ps. You should add a pipeline to solve the question that convert Image to tensor.
        """
        pic.convert('RGB')

        bg_idx = np.random.randint(0, self.size)
        #         self.imgs[bg_idx].show()
        #注意这要加self
        bg = self.getRectImage(self.imgs[bg_idx], self.weight, self.height)
        new_pic = self.gray2rgb(pic)
        return self.merge(new_pic, bg), self._get_pic_label(pic, label)

    def _get_pic_label(self, pic, label): # Image (28, 28) -> Narray(28, 28, 10)
        pic_arr = np.array(pic)
        new_shape = list(pic_arr.shape)
        new_shape.append(10)
        arr = np.zeros(new_shape, dtype=np.float32)


        label_arr = np.where(pic_arr[:, :]>0, 1, 0)
        arr[:,:, label] = label_arr
        # assert
        return torch.tensor(arr, dtype=torch.float32)

