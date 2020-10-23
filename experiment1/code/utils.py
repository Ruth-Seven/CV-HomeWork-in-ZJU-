import os
from image import Image
import cv2
import matplotlib.pyplot as plt
import pathlib

from config import *


# 载入文件夹中的图像
def load_images(path: pathlib.Path):
    images = []
    try:
        if path.is_file():
            raise
    except RuntimeError as e:
        Logger.debug(msg="空文件夹")


    # 递归搜索 ppm文件
    # or alternative choose: for cur_dir in path.iterdir():
    def get_image(pattern):
        for file in path.glob(pattern):
            images.append(Image(file))

    get_image("**/*.ppm")
    get_image("**/*.pgm")
    return images

# 载入一张图片
def load_image(path):
    return Image(path)

# 显示图片
def imshow(image):
    img = cv2.imread(image.filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


