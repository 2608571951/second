# coding: utf-8

# In[1]:


from __future__ import print_function
import os, random
from torchvision import transforms
import torch.optim as optim
from pytorchtools import EarlyStopping
import numpy as np
from FWNet import FWNet
from my_utils import *
import warnings

warnings.filterwarnings("ignore")


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:

GLOBAL_SEED = 1
BATCH_SIZE = 1
Net = None
DATA_PATH = './data_path_MR/'
MODEL_SAVE_PATH = './temp_model_MR_noKL'  # 去掉KL散度

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


# In[4]:


train_data = MyDataset(txt = DATA_PATH + 'train_list_whole_MR.txt',    # train_data是一个MyDataset对象
                       transform = transforms.ToTensor()) # 将numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之
### 数据加载
# dataset=train_data 调用MyDataset类中的__getitem__方法
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE,
                                          drop_last = True,
                                          num_workers = 2,
                                          shuffle = True,
                                          worker_init_fn = worker_init_fn)


# In[5]:


val_data = MyDataset(txt = DATA_PATH + 'val_list_whole_MR.txt',
                     transform = transforms.ToTensor())
valloader = torch.utils.data.DataLoader(dataset=val_data, batch_size = BATCH_SIZE,
                                        drop_last = True,
                                        num_workers = 2,
                                        shuffle = True,
                                        worker_init_fn = worker_init_fn)


# In[6]:


test_data = MyDataset(txt = DATA_PATH + 'test_list_whole_MR.txt',
                      transform = transforms.ToTensor())
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size = BATCH_SIZE,
                                         drop_last = True,
                                         num_workers = 2,
                                         shuffle = False,
                                         worker_init_fn = worker_init_fn)


# In[7]:

### 定义 网络，损失函数，优化器
print('定义网络，loss函数，优化器......')
net = FWNet(n_channels=1, n_classes=2)
net.to(device)
criterion = nn.MSELoss()  # 损失函数：最小均方误差
optimizer = optim.SGD(net.parameters(), lr = 0.03,
                      momentum = 0.9, weight_decay = 0.0005)  # 优化：随机梯度下降


# early_stopping = print_all_EarlyStopping(save_name = '../model_4/without-bg-{}'.format(TAG),
#                                          patience = 20, verbose = True)


if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

### 初始化 EarlyStopping 的对象
early_stopping = EarlyStopping(save_name = MODEL_SAVE_PATH + '/best_fusion_model_MR',
                                         patience = 10, verbose = True)


### 定义 训练次数
n_epochs = 30


### 定义 size=1*256*256，value=0.4 的张量
rho = torch.full(size=(BATCH_SIZE, 256, 256), fill_value=0.4, device = device)


### 计算损失
def calc_loss(ct, mr, img_fusion, outputs):
    """Mask MSE"""
    # bg_mask = get_bg_mask(ct, mr).type_as(outputs)  #  bg_mask中的值是0 或 1
    # fg_mask = (1.0 - bg_mask)
    # fg_mask[fg_mask < 0.4] = 0.4    #  fg_mask中的值是0.4 或 1
    # ### CT图像的输出与真实数据之间的差 + mr图像的输出与真实数据之间的差   L(reconstruct):用来衡量原图像和重构图像在像素水平的差异
    # loss = criterion(outputs[:, 0, :, :] * fg_mask, ct) + 7. * criterion(outputs[:, 1, :, :] * fg_mask, mr)
    # # loss = criterion(outputs[:, 0, :, :], ct) + 5. * criterion(outputs[:, 1, :, :], mr)


    """General MSE"""
    loss = criterion(outputs[:, 0, :, :], ct) + \
    criterion(outputs[:, 1, :, :], mr)


    """Sparsity Penalty （稀疏惩罚项）"""

    # sparsity_penalty = 3 * kl_divergence(rho, img_fusion)
    # loss += sparsity_penalty

    return loss


# In[8]:


## ===============Training==================
# net.train()和net.eval()两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数。
# 运用net.train()时，训练时每个min-batch时都会根据情况进行上述两个参数的相应调整，所有Batch Normalization的训练和测试时的操作不同。
# 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []
print('开始训练......')
for epoch in range(1, n_epochs + 1):
    net.train()

    # i 纪录的是训练样例有多少对
    for i, data in enumerate(iterable=trainloader, start=1): # trainloader 加载训练集数据
        ct, mr = data

        # 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        # 这句话需要写的次数等于需要保存GPU上的tensor变量的个数；一般情况下这些tensor变量都是最开始读数据时的tensor变量，后面衍生的变量自然也都在GPU上
        ct, mr = ct.to(device), mr.to(device)

        # 优化器梯度初始化为0
        optimizer.zero_grad()

        # # 调用 FWNet 的 forward 方法，进行一系列卷积池化等操作
        # torch.cat((ct, mr), dim = 1)：将ct,mr两个张量拼起来，dim=0:竖着拼；dim=1:横着拼
        img_fusion, outputs = net(torch.cat((ct, mr), dim=1))

        loss = calc_loss(ct, mr, img_fusion, outputs)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # evaluation
    net.eval()
    for i, data in enumerate(iterable=valloader, start=1):    # valloader 加载验证集数据
        ct, mr = data
        ct, mr = ct.to(device), mr.to(device)
        img_fusion, outputs = net(torch.cat((ct, mr), dim = 1))

        loss = calc_loss(ct, mr, img_fusion, outputs)

        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    early_stopping(valid_loss, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break

print('训练结束......')


# In[9]:
print('开始测试......')
# def test():
net.load_state_dict(torch.load(MODEL_SAVE_PATH + '/best_fusion_model_MR_checkpoint.pt'))


# In[13]:


all_data = MyDataset(txt = './data_path_MR/final_all_data_MR.txt',
                     transform = transforms.ToTensor())
allloader = torch.utils.data.DataLoader(all_data, batch_size = 1, num_workers = 1)


# In[14]:


SAVE_PATH = './my_result'

# generate result
RESULT_DIR = SAVE_PATH
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
net.eval()

#torch.no_grad() 是一个上下文管理器，被该语句内部的语句将不会计算梯度。
with torch.no_grad():
    for i, data in enumerate(allloader, 0):
        cases = i // 24
        img_id = i % 24
        img_ct, img_mr = data
        img_ct, img_mr = img_ct.to(device), img_mr.to(device)
        
        img_fusion, reconstruct_output = net(torch.cat((img_ct, img_mr),
                                                       dim = 1).to(device))
        # scale
        r = post_image(img_ct[0][0], img_mr[0][0], img_fusion[0][0]).cpu()
        io.imsave('{}/{}_{}.png'.format(RESULT_DIR, cases + 1, str(img_id).zfill(3)),
                np.array(r * 255, dtype = 'uint8'))
        print('save susseccs...')
print('测试结束......')
