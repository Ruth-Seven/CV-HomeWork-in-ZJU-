
import torch
from PIL import Image
import  numpy as np
from numpy import random
import torchsnooper



def sample_show(pics_1c, pics_3c, path):
    idx = [random.randint(0, min(len(pics_1c, pics_3c))) for _ in range(100)]
    biggerpic_show(pics_1c, idx, path / "before.png")
    biggerpic_show(pics_3c, idx, path / "after.png")

# get the 10 * 10 size picture
# @torchsnooper.snoop()
def biggerpic_show(pics, pic_idx, path):
    """
    @argument
    pics: list of Images or tensor and ndarray which shape is (batch, w, h, channel)
    pic_idx:a list of a hundrend numbers
    path: the path of the picture saved
    """

    if type(pics[0]) == Image.Image:
        shape = pics[0].size
    else:
        shape = pics[0].shape[-3:-1]
    shape = (shape[1], shape[0])    # change shape to numpy-style shape
    new_shape = [x * 10 for x in shape] #
    new_shape.append(3)
    real_shape = np.array(pics[0]).shape
    # for 1 channel pictures, resize to (height, weight, 1)
    if len(real_shape) == 2:
        real_shape = (real_shape[0], real_shape[1], 1)

    pic_arr = np.zeros(new_shape, dtype=np.float32)
    for i in range(10):
        for j in range(10):
            idx = pic_idx[i * 10 + j]
            arr = np.array(pics[idx])

            if real_shape[-1] > 3:
                max_arr = arr.argmax(axis=-1)

            # choose different assignment
            if real_shape[-1] == 3:
                pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1], :] = arr
            elif real_shape[-1] == 2:
                pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1],0:2] = arr
            elif real_shape[-1] == 1:
                for k in range(3):
                    pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1], k] = arr[:, :]
            elif real_shape[-1] > 3:
                # 3 * 3 * 2 种颜色分配
                list_bgb = np.array([(x, y, z)   for x in np.linspace(0, 255, 3) for y in np.linspace(0, 255, 3) for z in np.linspace(0, 255, 2)])
                temp = list_bgb[max_arr] #.transpose(1, 2, 0)
                assert temp.shape == (28, 28, 3)
                pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1],:] = temp
                # start_ch1 = real_shape[-1] // 3
                # start_ch2 = start_ch1 * 2
                # gap1 = 255 / (start_ch1 - 0 - 1)
                # weight = 0
                # for k in range(start_ch1):
                #     pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1], 0:1] \
                #         += arr[:, :, k: k + 1] / 255 * weight
                #     weight += gap1
                # weight = 0
                #
                # gap2 = 255 / (start_ch2 - start_ch1 - 1)
                # weight = 0
                # for k in range(start_ch1, start_ch2):
                #     pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1], 1:2] \
                #         += arr[:, :, k: k + 1] / 255 * weight
                #     weight += gap2
                # weight = 0
                #
                # gap3 = 255 / (real_shape[-1] - start_ch1 - 1)
                # weight = 0
                # for k in range(start_ch2, real_shape[-1]):
                #     pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1], 2:3] \
                #         += arr[:, :, k: k + 1] / 255 * weight
                #     weight += gap3

    img = Image.fromarray(pic_arr.astype(np.uint8))
    img.save(path)
    print(f"已保存图片{path.name}")
# @torchsnooper.snoop()
def show_pic_11c(arr, l=0):
    shape = arr.shape
    if shape[0] == shape[1]: #
        arr = arr.permute(2,0,1)
        shape = arr.shape

    if type(arr) == torch.Tensor:
        arr = arr.cpu().detach().numpy()
    arr.astype(np.uint8)
    cat_arr = arr.reshape(shape[0] * shape[2], shape[1] ) #reshape的综合应用
    Image.fromarray(cat_arr).show()
    Image.fromarray(arr[l, :,:]).show()


# def biggerpic_show_1c(data, pic_idx, path):
#     shape = data[0].size
#     new_shape = [x * 10 for x in shape]
#     pic_arr = np.zeros(new_shape, dtype=np.uint8)
#     for i in range(10):
#         for j in range(10):
#             idx = pic_idx[i * 10 + j]
#             arr = np.array(data[idx][0])
#             pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]] = arr
#     img = Image.fromarray(pic_arr)
#     img.save(path)
#
# def biggerpic_show_1c(data, pic_idx, path):
#     shape = data[0].size
#     new_shape = [x * 10 for x in shape]
#     pic_arr = np.zeros(new_shape, dtype=np.uint8)
#     for i in range(10):
#         for j in range(10):
#             idx = pic_idx[i * 10 + j]
#             arr = np.array(data[idx][0])
#             pic_arr[i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]] = arr
#     img = Image.fromarray(pic_arr)
#     img.save(path)


if __name__ == "__main__":
    test = torch.rand(4,28,28) * 255
    test[:,14:17,:] = 255
    show_pic_11c(test)


