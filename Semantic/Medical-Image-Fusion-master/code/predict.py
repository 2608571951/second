"""
This script can load our trained model to generate a fused image.
Call example: python3 predict.py ct_img_path mr_img_path
"""

import sys
from torchvision import transforms
import numpy as np
from FWNet import FWNet
from my_utils import *
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = FWNet(1, 2)
net.to(device)
#net.load_state_dict(torch.load('./model/fusion_model.pt'))
net.load_state_dict(torch.load('./temp_model_MR_noKL/best_fusion_model_MR_checkpoint.pt'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    ]
)

# img_ct = io.imread(sys.argv[1]) # '../test/case10/ct1_012.gif'
# img_mr = io.imread(sys.argv[2]) # '../test/case10/mr2_012.gif'
for i in range(18):
    img_ct = io.imread('../test_img/ct/ct_' + str(i) + '.png') # '../test/case10/ct1_012.gif'
    img_mr = io.imread('../test_img/mr/mr_' + str(i) + '.png') # '../test/case10/mr2_012.gif'
    img_ct = transform(img_ct)
    img_mr = transform(img_mr)
    img_ct = torch.unsqueeze(img_ct, 1)
    img_mr = torch.unsqueeze(img_mr, 1)

    img_fusion, oo = net(torch.cat((img_ct, img_mr), dim = 1).to(device))
    oo = oo.cpu()
    img_fusion = img_fusion.cpu()
    img_fusion_post = post_image(img_ct[0][0], img_mr[0][0], img_fusion[0][0], chg_bg = True)

    #torch.no_grad() 是一个上下文管理器，被该语句内部的语句将不会计算梯度。
    with torch.no_grad():
        io.imsave('./test_result_noKL/fusion_result_' + str(i) + '.png', np.array(img_fusion_post * 255, dtype = 'uint8'))
