from unet_parts import *

class FWNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FWNet, self).__init__()
        
        self.inc_gen = inconv(2, 32)              # x1 256
        self.down1_gen = down(32, 64)             # x2 128
        self.down2_gen = down(64, 128)            # x3 64
        self.down3_gen = down(128, 256)           # x4 32
        self.down4_gen = down(256, 256)           # x5 16
        self.up1_gen = up(512, 128)
        self.up2_gen = up(256, 64)
        self.up3_gen = up(128, 32)
        self.up4_gen = up(64, 32)
        self.outc_gen = outconv(32, 1)


        
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)
        
        self.relu = nn.ReLU(inplace = True)



    def forward(self, x):
        # generate network
        x1 = self.inc_gen(x)
        # print('x1.shape:',x1.shape)
        x2 = self.down1_gen(x1)
        # print('x2.shape:', x2.shape)
        x3 = self.down2_gen(x2)
        # print('x3.shape:', x3.shape)
        x4 = self.down3_gen(x3)
        # print('x4.shape:', x4.shape)
        x5 = self.down4_gen(x4)
        # print('x5.shape:', x5.shape)
        x = self.up1_gen(x5, x4)
        # print('x6.shape:', x.shape)
        x = self.up2_gen(x, x3)
        # print('x7.shape:', x.shape)
        x = self.up3_gen(x, x2)
        # print('x8.shape:', x.shape)
        x = self.up4_gen(x, x1)
        # print('x9.shape:', x.shape)
        x = self.outc_gen(x)
        # print('x10.shape:', x.shape)
        fr = F.sigmoid(x)
        # print('fr.shape:', fr.shape)
        
        # rebuild network
        x1 = self.inc(fr)
        # print('re_x1.shape:', x1.shape)
        x2 = self.down1(x1)
        # print('re_x2.shape:', x2.shape)
        x3 = self.down2(x2)
        # print('re_x3.shape:', x3.shape)
        x4 = self.down3(x3)
        # print('re_x4.shape:', x4.shape)
        x5 = self.down4(x4)
        # print('re_x5.shape:', x5.shape)
        x = self.up1(x5, x4)
        # print('re_x6.shape:', x.shape)
        x = self.up2(x, x3)
        # print('re_x7.shape:', x.shape)
        x = self.up3(x, x2)
        # print('re_x8.shape:', x.shape)
        x = self.up4(x, x1)
        # print('re_x9.shape:', x.shape)
        x = self.outc(x)
        # print('re_x10.shape:', x.shape)
        # print('fr2.shape:', F.sigmoid(x).shape)
        return fr, F.sigmoid(x)
