import pdb

import numpy as np
import os, math
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob

from basicsr.models.archs.uformer_OriKV_Qsigmoid_arch import Uformer_OriKV_Qsigmoid
from skimage import img_as_ubyte

def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask

# python test.py --input_dir /root/autodl-tmp/Omni-Deblurring/Motion_Deblurring/Datasets/ --weights /root/autodl-tmp/Omni-Deblurring/Motion_Deblurring/pretrained_models/net_g_latest.pth --dataset GoPro

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Restormer-lrg/Motion_Deblurring/Datasets/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./Restormer-lrg/Motion_Deblurring/pretrained_models/net_g_latest.pth', type=str,
                    help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset')  # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

args = parser.parse_args()
# pdb.set_trace()
####### Load yaml #######
yaml_file = 'Options/Deblurring_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Uformer_OriKV_Qsigmoid()

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
#pdb.set_trace()
factor = 8
dataset = args.dataset
result_dir = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        # .set_trace()
        img = np.float32(utils.load_img(file_)) / 255.  # img[H,W,C]
        img = torch.from_numpy(img).permute(2, 0, 1)  # img[C,H,W]
        input_ = img.unsqueeze(0).cuda()  # img[B,C,H,W]

        # Padding in case images are not multiples of 8
        b, c, h, w = input_.shape[0], input_.shape[1], input_.shape[2], input_.shape[3]
        tile = 256
        tile_overlap = 32
        scale = 1
        tile = min(tile, h, w)

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * scale, w * scale).type_as(input_)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model_restoration(in_patch)
                if isinstance(out_patch, list):
                    out_patch = out_patch[-1]
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(out_patch)
                W[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(out_patch_mask)
        restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()  # B,H,W,C

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                       img_as_ubyte(restored))

