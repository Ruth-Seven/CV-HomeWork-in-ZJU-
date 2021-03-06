import torch
import time
from datetime import timedelta
def flatted_tensor_size(x:torch.tensor)-> int:
    ans = 1
    for dim in x.shape[1:]:
        ans *= dim
    return dim

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


## decorator method

def cost_time(func):
    def wrap_cost_time(*args, **kwargs):
        start_time = time.time()
        print(f"_______RUNING {func.__class__}_______")
        res = func(*args, **kwargs)
        diff_time = time.time() - start_time
        print(f"_______Cost {timedelta(seconds=int(round(diff_time)))}time")
        return res
    return wrap_cost_time


## data transforms
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])


## visualization

# dramatically show  changed log of loss function
def loss_windwon(config, viz, loss_list, cur_batch_win):
    if viz.check_connection():
        cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor([ range(len(loss_list))]),
                                win=cur_batch_win, name='current_batch_loss',
                                update=(None if cur_batch_win is None else 'replace'),
                                opts=config.cur_batch_win_opts)

if __name__ == "__main__":
    ones = torch.ones(2,2,device='cuda')
    print(flatted_tensor_size(ones))
