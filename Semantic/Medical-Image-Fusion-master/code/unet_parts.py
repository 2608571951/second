import torch
import torch.nn as nn
import torch.nn.functional as F



## 定义双层卷积
class double_conv(nn.Module):
    '''(conv => BN(BatchNorm) => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        ## torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中。另外，也可以传入一个有序模块。
        ## nn.BatchNormal2d(num_featres):在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
         # 返回一个shape与num_features相同的tensor
         # num_features为输入batch中图像的channle数（按每一个channle来做归一化）
        ## nn.ReLU(inplace) 当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding = 1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding = 1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
## 双层卷积操作
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x





## 下采样 ： 最大池化操作，双层卷积操作
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x





## 上采样 : 上采样操作，双层卷积操作
 ## nn.Upsample(): 参数scale_factor  指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
 # 参数mode  可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. 默认使用'nearest'
 # 参数align_corners  如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。仅当使用的算法为'linear', 'bilinear'or 'trilinear'时可以使用。默认设置为False
 ## nn.ConvTranspose2d() : 反卷积操作
  # 符号//   整除（向下取整）
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear = True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch//2, out_channels=in_ch//2, kernel_size=2, stride = 2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is BCHW（batch_size, 通道数，高度，宽度）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
#         print(diffY, diffX)

        ## pad表示：左填充，右填充，上填充，下填充
        x1 = F.pad(input=x1, pad=(diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        

        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x










## 应该是全卷积操作
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

