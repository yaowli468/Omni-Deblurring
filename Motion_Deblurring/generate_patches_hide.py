##### Data preparation file for training Restormer on the GoPro Dataset ########
import pdb

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing

def train_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int))  # np.int换成int
        # 0起点，w-patch_size终点，patch_size-overlap是步长
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename = os.path.join(hr_tar, filename + '.png')
        
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

def val_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    lr_savename = os.path.join(lr_tar, filename + '.png')
    hr_savename = os.path.join(hr_tar, filename + '.png')

    w, h = lr_img.shape[:2]

    i = (w-val_patch_size)//2
    j = (h-val_patch_size)//2
                
    lr_patch = lr_img[i:i+val_patch_size, j:j+val_patch_size,:]
    hr_patch = hr_img[i:i+val_patch_size, j:j+val_patch_size,:]

    cv2.imwrite(lr_savename, lr_patch)
    cv2.imwrite(hr_savename, hr_patch)

############ Prepare Training data ####################
num_cores = 10
patch_size = 512
overlap = 256  # 减小重叠，减少图像数量
p_max = 0

# src = 'Datasets/Downloads/GoPro'  # 源文件夹，训练文件夹，包含input和target两个文件夹
# tar = 'Datasets/train/GoPro'
src = '/root/autodl-tmp/hide_src/train'
tar = '/root/autodl-tmp/Datasets/train/HIDE'


lr_tar = os.path.join(tar, 'input_crops')  # 生成的目标文件夹
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.png')) + glob(os.path.join(src, 'input', '*.jpg')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.png')) + glob(os.path.join(src, 'target', '*.jpg')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


############ Prepare validation data ####################
val_patch_size = 256
# src = 'Datasets/test/GoPro'
# tar = 'Datasets/val/GoPro'
src = '/root/autodl-tmp/hide_src/test'
tar = '/root/autodl-tmp/Datasets/val/HIDE'


lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, 'input', '*.png')) + glob(os.path.join(src, 'input', '*.jpg')))
hr_files = natsorted(glob(os.path.join(src, 'target', '*.png')) + glob(os.path.join(src, 'target', '*.jpg')))

files = [(i, j) for i, j in zip(lr_files, hr_files)]

Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
