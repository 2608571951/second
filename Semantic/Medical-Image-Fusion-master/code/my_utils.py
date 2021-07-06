from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import torch

class MyDataset(Dataset):
    def __init__(self, txt, transform = None, target_transform = None):
        super(MyDataset,self).__init__()
        lists = open(txt, 'r')
        imgs = []
        for line in lists:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))   # 图片的本地地址

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        ct, mr = self.imgs[index]
        img_ct = io.imread(ct)
        img_mr = io.imread(mr)
        if self.transform is not None:
            img_ct = self.transform(img_ct)      #将读取的图片转换成形状为(C,H, W)的Tensor格式，且[0，255]归一化到[0 ，1.0]之间
            img_mr = self.transform(img_mr)
        return img_ct, img_mr

    def __len__(self):
        return len(self.imgs)

# If the pixel value is below the threshold, this pixel is the background. 
# Otherwise, it's not.
BG_THRESHOLD = 17.0/256


def get_bg_mask(ct, mr):
    bg_mask = ((ct <= BG_THRESHOLD) & (mr <= BG_THRESHOLD))
    return bg_mask

def get_fg_mask(ct, mr):
    fg_mask = ((ct > BG_THRESHOLD) | (mr > BG_THRESHOLD))
    return fg_mask

def scale_image(img, fg_mask = None):
    if fg_mask is None:
        fg_mask = torch.ones(img.shape).type(torch.ByteTensor)
    fg_vals = torch.masked_select(img, fg_mask)
    minv = fg_vals.min()
    maxv = fg_vals.max()
    img = (img - minv)
    img = img / (maxv - minv)
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def post_image(ct, mr, fused, chg_bg = True, inverse = True):
    bg_mask = get_bg_mask(ct, mr)
    fg_mask = get_fg_mask(ct, mr)
    r = scale_image(fused, fg_mask)
    if inverse:
        r = 1 - r
    if chg_bg: 
        r[bg_mask] = 0
    return r

def kl_divergence(p, q):    #  rho , img_fusion
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = p.reshape(-1, )
    q = q.reshape(-1, )
    # 返回结果是一个与输入维度dim相同的张量，每个元素的取值范围在（0,1）区间
    # dim:指明维度，dim=0表示按列计算；dim=1表示按行计算
    p = F.softmax(p, dim = 0)
    q = F.softmax(q, dim = 0)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2
