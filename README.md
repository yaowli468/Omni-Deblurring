<!-- Title -->
## Omni-Deblurring: Capturing Omni-Range Context for Image Deblurring
### Note: uformer_OriKV_Qsigmoid_arch.py will be published upon acceptance of the paper

![image text](https://github.com/yaowli468/Omni-Deblurring/blob/main/IMG/Framework.jpg)
 
## Dependencies
* Linux(Tested on Ubuntu 18.04) 
* Python 3.8 (Recomend to use [Anaconda](https://www.anaconda.com/products/individual#linux))
* Pytorch 1.11
* tqdm==4.66.4
* opencv-python==4.10.0.84
* natsort==8.4.0
* lmdb==1.5.1
* timm==1.0.7
* einops==0.8.0
* scipy==1.10.1
* scikit-image==0.21.0
* tensorboard==2.14.0

## Get Started

### Download
* Pretrained model can be downloaded from [HERE](https://pan.baidu.com/s/1buNU5yv4vWTXi9Pw5BocJA)(i71k), please put them to './Motion_Deblurring/pretrained_models/'

### Testing
1. Follow the instructions below to test our model. Please modify the file path, e.g. `--input_dir`, `--weights`. Select dataset, e.g. `--dataset` GoPro or HIDE.
    ```sh
    python test.py --input_dir ./Motion_Deblurring/Datasets/ --weights ./Motion_Deblurring/pretrained_models/net_g_latest.pth --dataset GoPro
    ```

### Training
1. Generate image patches from full-resolution training images of GoPro dataset.
   ```sh  
   python generate_patches_gopro.py 
   ```
2. Modify the file `./Motion_Deblurring/Options/Deblurring_Restormer.yml`.

3. Follow the instructions below to train our model.
   ```sh  
   python setup.py develop --no_cuda_ext
   python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro.yml --launcher pytorch
   ```
## Acknowledgments
This code is based on [Restormer](https://github.com/swz30/Restormer). Thanks for their greate works.

 



