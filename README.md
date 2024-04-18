<div align="center">
<h1 align="center">ChangeMamba</h1>

<h3>ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model</h3>


[Hongruixuan Chen](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1 #</sup>, [Jian Song](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,2 #</sup>, [Chengxi Han](https://chengxihan.github.io/)<sup>3</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>2</sup>, [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,2 *</sup>

<sup>1</sup> The University of Tokyo, <sup>2</sup> RIKEN AIP,  <sup>3</sup> Wuhan University.

<sup>#</sup> Equal contribution, <sup>*</sup> Corresponding author

**Paper: ([arXiv 2404.03425](https://arxiv.org/pdf/2404.03425.pdf))** 

[**Overview**](#overview) | [**Get Started**](#%EF%B8%8Flets-get-started) | [**Main Results**](#%EF%B8%8Fmain-results) | [**Reference**](#reference) | [**Q & A**](#q--a)

</div>

## 🛎️Updates
* **` Notice🐍🐍`**: The code of this repo has been updated! Some of the retrained model weights have been uploaded for usage! We'd appreciate it if you could give this repo ⭐️ and stay tuned!
* **` April 18th, 2024`**: The retrained weight of MambaBCD-Base on the [WHU-CD](https://drive.google.com/file/d/1K7aSuT3os7LR9rUvoyVNP-x0hWKZocrn/view?usp=drive_link) (F1 score ***94.19%***) is now avaiable. You are welcome to use it!!
* **` April 17th, 2024`**: The retrained weight of MambaBDA-Tiny on the [xBD](https://drive.google.com/file/d/11UrVyntxPDFf1Qt0TlDORoh4eoM7WJsS/view?usp=drive_link) (oaF1 score ***81.11%***) is now avaiable. You are welcome to use it!!
* **` April 15th, 2024`**: The retrained weights of MambaBCD-Small on the [SYSU](https://drive.google.com/file/d/1ZEPF6CvvFynL-yu_wpEYdpHMHl7tahpH/view?usp=drive_link) (F1 score ***83.36%***), Mamba-BCD-Tiny on the [LEVIR-CD+](https://drive.google.com/file/d/1AtiXBBCoofi1e5g4STYUzBgJ1fYN4VhN/view?usp=drive_link) (F1 score of ***88.03%***) and [WHU-CD](https://drive.google.com/file/d/1ZLKXhGKgnWoyS0X8g3HS45a3X1MP_QE6/view?usp=drive_link) with (F1 score ***94.09%***) are now avaiable. You are welcome to use them!!
* **` April 12th, 2024`**: The new [[arXiv](https://arxiv.org/pdf/2404.03425.pdf)] version containing new accuracy and more experiments is now online! The weights for different models will be released soon!
* **` April 05th, 2024`**: The [[arXiv](https://arxiv.org/pdf/2404.03425.pdf)] version is online!
* **` April 05th, 2024`**: The models and training code for MambaBCD, MambaSCD, and MambaBDA have been organized and uploaded. You are welcome to use them!!

## 🔭Overview

* [**ChangeMamba**](https://arxiv.org/pdf/2404.03425.pdf) serves as a strong benchmark for change detection tasks, including binary change detection (MambaBCD), semantic change detection (MambaSCD), and building damage assessment (MambaBDA). 

<p align="center">
  <img src="figures/network_architecture.png" alt="accuracy" width="90%">
</p>

* **Spatio-temporal relationship learning methods of ChangeMamba**

<p align="center">
  <img src="figures/STLM.png" alt="arch" width="60%">
</p>


## 🗝️Let's Get Started!
### A. Installation
The repo is based on the [VMama repo](https://github.com/MzeroMiko/VMamba), thus you need to install it first. The following installation sequence is taken from the VMamba repo. Also, note that the code in this repo runs under Linux system. We have not tested whether it works under other OS.

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/ChenHongruixuan/MambaCD.git
cd MambaCD
```


**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n changemamba
conda activate changemamba
```

***Install dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```


***Dependencies for `Detection` and `Segmentation` (optional in VMamba)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```
### B. Download Pretrained Weight
Also, please download the pretrained weights of [VMamba-Tiny](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230_ckpt_epoch_262.pth), [VMamba-Small](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_small_0229_ckpt_epoch_222.pth), and [VMamba-Base](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229_ckpt_epoch_237.pth) and put them under 
```bash
project_path/MambaCD/pretrained_weight/
```

### C. Data Preparation
***Binary change detection***

The three datasets [SYSU](https://github.com/liumency/SYSU-CD), [LEVIR-CD+](https://chenhao.in/LEVIR/) and [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) are used for binary change detection experiments. Please download them and make them have the following folder/file structure:
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SYSU
├── train
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   └── GT
│       ├──00001.png 
│       ...   
│   
├── test
│   ├── ...
│   ...
│  
├── train.txt   # Data name list, recording all the names of training data
└── test.txt    # Data name list, recording all the names of testing data
```

***Semantic change detection***

The [SECOND dataset](https://captain-whu.github.io/SCD/) is used for semantic change detection experiments. Please download it and make it have the following folder/file structure:

```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SECOND
├── train
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   ├── GT_CD   # Binary change map
│   │   ├──00001.png 
│   │   ... 
│   │
│   ├── GT_T1   # Land-cover map of T1
│   │   ├──00001.png 
│   │   ...  
│   │
│   └── GT_T2   # Land-cover map of T2
│       ├──00001.png 
│       ...  
│   
├── test
│   ├── ...
│   ...
│ 
├── train.txt
└── test.txt
```

***Building damage assessment***

The xBD dataset can be downloaded from [xView 2 Challenge website](https://xview2.org/dataset). After downloading it, please organize it into the following structure: 
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/xBD
├── train
│   ├── images
│   │   ├──guatemala-volcano_00000000_pre_disaster.png
│   │   ├──guatemala-volcano_00000000_post_disaster.png
│   │   ...
│   │
│   └── masks
│       ├──guatemala-volcano_00000003_pre_disaster.png
│       ├──guatemala-volcano_00000003_post_disaster.png
│       ... 
│   
├── test
│   ├── ...
│   ...
│
├── holdout
│   ├── ...
│   ...
│
├── train.txt # Data name list, recording all the names of training data
├── test.txt  # Data name list, recording all the names of testing data
└── holdout.txt  # Data name list, recording all the names of holdout data
```


### D. Model Training
Before training models, please enter into [**changedetection**] folder, which contains all the code for network definitions, training and testing. 

```bash
cd <project_path>/MambaCD/changedetection
```

***Binary change detection***

The following commands show how to train and evaluate MambaBCD-Small on the SYSU dataset:
```bash
python script/train_MambaBCD.py  --dataset 'SYSU' \
                                 --batch_size 16 \
                                 --crop_size 256 \
                                 --max_iters 320000 \
                                 --model_type MambaBCD_Small \
                                 --model_param_path '<project_path>/MambaCD/changedetection/saved_models' \ 
                                 --train_dataset_path '<dataset_path>/SYSU/train' \
                                 --train_data_list_path '<dataset_path>/SYSU/train_list.txt' \
                                 --test_dataset_path '<dataset_path>/SYSU/test' \
                                 --test_data_list_path '<dataset_path>/SYSU/test_list.txt'
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
```

***Semantic change detection***

The following commands show how to train and evaluate MambaSCD-Small on the SECOND dataset:
```bash
python script/train_MambaSCD.py  --dataset 'SECOND' \
                                 --batch_size 16 \
                                 --crop_size 256 \
                                 --max_iters 800000 \
                                 --model_type MambaSCD_Small \
                                 --model_param_path '<project_path>/MambaCD/changedetection/saved_models' \ 
                                 --train_dataset_path '<dataset_path>/SECOND/train' \
                                 --train_data_list_path '<dataset_path>/SECOND/train_list.txt' \
                                 --test_dataset_path '<dataset_path>/SECOND/test' \
                                 --test_data_list_path '<dataset_path>/SECOND/test_list.txt'
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
```

***Building Damge Assessment***

The following commands show how to train and evaluate MambaBDA-Small on the xBD dataset:
```bash
python script/train_MambaSCD.py  --dataset 'xBD' \
                                 --batch_size 16 \
                                 --crop_size 256 \
                                 --max_iters 800000 \
                                 --model_type MambaBDA_Small \
                                 --model_param_path '<project_path>/MambaCD/changedetection/saved_models' \ 
                                 --train_dataset_path '<dataset_path>/xBD/train' \
                                 --train_data_list_path '<dataset_path>/xBD/train_list.txt' \
                                 --test_dataset_path '<dataset_path>/xBD/test' \
                                 --test_data_list_path '<dataset_path>/xBD/test_list.txt'
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
```
### E. Inference Using Our Weights

Before inference, please enter into [**changedetection**] folder. 
```bash
cd <project_path>/MambaCD/changedetection
```

***Binary change detection***

The following commands show how to infer binary change maps using trained MambaBCD-Tiny on the LEVIR-CD+ dataset:
```bash
python script/infer_MambaBCD.py  --dataset 'LEVIR-CD+' \
                                 --model_type 'MambaBCD_Tiny' \
                                 --test_dataset_path '<dataset_path>/LEVIR-CD+/test' \
                                 --test_data_list_path '<dataset_path>/LEVIR-CD+/test_list.txt' \
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'
                                 --resume '<saved_model_path>/MambaBCD_Tiny_LEVIRCD+_F1_0.8803.pth'
```



## ⚗️Main Results


* *The encoders for all the above ChangeMamba models are the the VMamba architecture initialized with ImageNet pre-trained weight.*

* *Some of comparison methods are not open-sourced. Their accuracy and number of parameters are obtained based on our own implementation.*


### **Binary Change Detection on SYSU**

| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 87.49 | 73.14 | 57.66 | 64.99 | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) | 97.83 | 74.70  | 59.62  |  73.57  | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) |  89.59 | 78.80 | 65.02 | 71.92  |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 90.77 | 80.44 | 67.28 | 74.40 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |    89.52 | 77.41 | 63.14 | 70.59 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  91.19 | 79.92 | 66.55 | 74.31 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  90.88 | 80.09 | 66.79  | 74.18 | 66.79  | 74.18| -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  91.51 | 81.58 | 68.89  | 76.06| 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) | 90.12 | 78.81 | 65.03 | 72.37  | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  90.76 | 79.41 | 65.84 | 73.47 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny | 91.36 | 81.29 | 68.48	| 75.68 | 17.13 | 45.74 | -- |
| MambaBCD-Small | 92.39  | 83.36 | 71.46 | 78.43 | 49.94 | 114.82 | [[GDrive](https://drive.google.com/file/d/1ZEPF6CvvFynL-yu_wpEYdpHMHl7tahpH/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1f8iwuKCkElU9rc24_ZzXBw?pwd=46p5)] |
| MambaBCD-Base |  92.30 | 83.11 | 71.10 | 78.13 | 84.70 | 179.32 | -- |

### **Binary Change Detection on LEVIR-CD+**
| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 97.54   | 70.42 |  54.34  | 69.14  | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) | 97.83 | 74.70  | 59.62  |  73.57  | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) | 98.70 | 84.07 | 72.52 | 83.39  |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 98.67 | 83.20 |  71.23  | 82.50 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |   98.22 | 77.56  |  63.34 |  76.63 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  98.63 |  83.68 |  71.94  |  82.97 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  98.66 | 83.63 | 71.86 | 82.93 | 28.37 | 244.54 | -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  98.92 | 85.60 | 74.82 | 84.98| 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) |  98.01 | 75.87 | 61.12 | 74.83  | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  98.60 | 82.53 | 70.26 | 81.80 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny | 99.03 | 88.04 | 78.63 | 87.53 | 17.13 | 45.74 | [[GDrive](https://drive.google.com/file/d/1AtiXBBCoofi1e5g4STYUzBgJ1fYN4VhN/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/13dGC_J-wyIfoPwoPJ5Uc6Q?pwd=8ali)] |
| MambaBCD-Small | 99.02 | 87.81 | 78.27 | 87.30 | 49.94 | 114.82 | -- |
| MambaBCD-Base |  99.06 | 88.39 | 79.20 | 87.91 | 84.70 | 179.32 | -- |

### **Binary Change Detection on WHU-CD**
| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 98.87  | 84.89  | 73.74  | 84.30 | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) |  99.10  | 87.70 | 78.09 |  87.23 | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) | 99.31  |  89.91| 81.67| 89.56 |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 99.19 | 89.10 | 80.34 | 88.68 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |  99.16 | 88.16 | 78.82 | 87.72 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  99.48 | 92.59 | 86.21 | 92.33 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  99.09 | 87.79 | 78.44 | 87.44 | 28.37 | 244.54 | -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  99.50 | 93.04 | 87.00 | 92.78 | 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) |  99.10 | 87.39 | 77.61 | 86.93 | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  99.27 | 90.04 | 81.88 | 89.66 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny |  99.57 | 94.09 | 88.84 | 93.87 | 17.13 | 45.74 | [[GDrive](https://drive.google.com/file/d/1ZLKXhGKgnWoyS0X8g3HS45a3X1MP_QE6/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1DhTedGZdIC80y06tog1xbg?pwd=raf0)] |
| MambaBCD-Small |  99.57 | 94.06 | 88.79 | 93.84 | 49.94 | 114.82 | -- |
| MambaBCD-Base |  99.58 | 94.19 | 89.02 | 93.98 | 84.70 | 179.32 | [[GDrive]](https://drive.google.com/file/d/1K7aSuT3os7LR9rUvoyVNP-x0hWKZocrn/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1o6Z6ecIJ59K9eB2KqNMD9w?pwd=4mqd)] |


### **Semantic Change Detection on SECOND**
| Method |  Overall Accuracy | F1 Score | IoU | SeK | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [HRSCD-S1](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300992) | 45.77 | 38.44 | 62.72   | 5.90   | 3.36   | 8.02 | -- |
| [HRSCD-S2](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300992) | 85.49 | 49.22  |  64.43  | 10.69 | 6.39 |  14.29 | -- |
| [HRSCD-S3](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300992) | 84.62 |  51.62  | 66.33  | 11.97|  12.77|   42.67 | -- |
| [HRSCD-S4](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300992) | 86.62   |58.21  | 71.15  |  18.80 | 13.71 |  43.69| -- |
| [ChangeMask](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002835) | 86.93  | 59.74  | 71.46 |  19.50 | 2.97 | 37.16 | -- |
| [SSCD-1](https://github.com/ggsDing/Bi-SRNet) | 87.19 | 61.22 |  72.60 |  21.86 |  23.39|  189.91 | -- |
| [Bi-SRNet](https://github.com/ggsDing/Bi-SRNet) |   87.84 | 62.61  | 73.41  | 23.22 | 19.44  | 63.72 | -- |
| [TED](https://github.com/ggsDing/SCanNet) | 87.39  | 60.34   | 72.79  | 22.17 | 42.16 | 75.79  | -- |
| [SMNet](https://www.mdpi.com/2072-4292/15/4/949) | 86.68 | 60.34   | 71.95 |  20.29 | 19.44  | 63.72 | -- |
| [ScanNet](https://github.com/ggsDing/SCanNet) | 87.86  | 63.66 |  73.42  | 23.94 | 27.90 | 264.95  | -- |
| MambaSCD-Tiny |  88.07  |  63.44  |  73.33  | 23.34 | 21.51 | 73.42 | -- |
| MambaSCD-Small | 88.38  | 64.10  | 73.61  | 24.04 | 54.28  |  146.70 | -- |
| MambaSCD-Base | 88.12  |  64.03  |  73.68   | 24.11 | 89.99  | 211.55 | -- |



### **Building Damage Assessment on xBD**
| Method |  F1_loc | F1_clf  | F1_oa |  Param | GFLOPs | ckpts
| :---: | :---: | :---: |  :---: |  :---: | :---: | :---: | 
| [xView2 Baseline](https://github.com/DIUx-xView/xView2_baseline) | 80.47 | 3.42 | 26.54 | -- | -- | -- |
| [Siamese-UNet](https://github.com/vdurnov/xview2_1st_place_solution) | 85.92  | 65.58  | 71.68  |  -- | -- | -- |
| [MTF](https://github.com/ethanweber/xview2) |  83.60 | 70.02 | 74.10 | -- | -- | -- |
| [ChangeOS-101](https://github.com/Z-Zheng/ChangeOS) |  85.69 | 71.14 | 75.50 | -- | -- | -- |
| [ChangeOS-101-PPS](https://github.com/Z-Zheng/ChangeOS) |  85.69 | 75.44 | 78.52  | -- | -- | -- |
| [DamFormer](https://arxiv.org/abs/2201.10953) |  86.86 |72.81 |77.02| -- | -- | -- |
| MambaBDA-Tiny |  87.53 | 78.35  | 81.11  | 19.74 | 59.57 | [[GDrive](https://drive.google.com/file/d/11UrVyntxPDFf1Qt0TlDORoh4eoM7WJsS/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/19r9lXXuwkeepfPTpj77IFg?pwd=u5a9)] |
| MambaBDA-Small | 86.61 | 78.80 | 81.14 | 52.11 |  130.80 | -- |
| MambaBDA-Base | 87.38 | 78.84| 81.41 | 87.76 | 195.43 | -- |


## 📜Reference

If this code or dataset contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@article{chen2024changemamba,
      title={ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model}, 
      author={Hongruixuan Chen and Jian Song and Chengxi Han and Junshi Xia and Naoto Yokoya},
      year={2024},
      eprint={2404.03425},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```



## 🤝Acknowledgments
This project is based on VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ScanNet ([paper](https://arxiv.org/abs/2212.05245), [code](https://github.com/ggsDing/SCanNet)), xView2 Challenge ([paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf), [code](https://github.com/DIUx-xView/xView2_baseline)). Thanks for their excellent works!!

## 🙋Q & A
***For any questions, please feel free to [contact us.](mailto:Qschrx@gmail.com)***
