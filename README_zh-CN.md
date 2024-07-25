<div align="center">
<h1 align="center">ChangeMamba</h1>

<h3>ChangeMamba: 利用时空状态空间模型进行遥感影像变化检测</h3>


[陈洪瑞轩](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1 #</sup>, [宋健](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,2 #</sup>, [韩承熙](https://chengxihan.github.io/)<sup>3</sup>, [夏俊士](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>2</sup>, [横矢直人](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,2 *</sup>

<sup>1</sup> 东京大学, <sup>2</sup> 理化学研究所先进智能研究中心,  <sup>3</sup> 武汉大学.

<sup>#</sup> 共同第一作者, <sup>*</sup> 通讯作者

[![TGRS paper](https://img.shields.io/badge/TGRS-paper-00629B.svg)](https://ieeexplore.ieee.org/document/10565926)  [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2404.03425.pdf)

[**简介**](#简介) | [**开始使用**](#%EF%B8%8F开始使用) | [**结果下载**](#%EF%B8%8F结果下载) | [**常见问题**](#引用) | [**其他**](#联系我们) | [**English Version**](https://github.com/ChenHongruixuan/MambaCD)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/changemamba-remote-sensing-change-detection/change-detection-on-sysu-cd)](https://paperswithcode.com/sota/change-detection-on-sysu-cd?p=changemamba-remote-sensing-change-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/changemamba-remote-sensing-change-detection/change-detection-on-levir)](https://paperswithcode.com/sota/change-detection-on-levir?p=changemamba-remote-sensing-change-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/changemamba-remote-sensing-change-detection/change-detection-on-whu-cd)](https://paperswithcode.com/sota/change-detection-on-whu-cd?p=changemamba-remote-sensing-change-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/changemamba-remote-sensing-change-detection/change-detection-on-second)](https://paperswithcode.com/sota/change-detection-on-second?p=changemamba-remote-sensing-change-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/changemamba-remote-sensing-change-detection/2d-semantic-segmentation-on-xbd)](https://paperswithcode.com/sota/2d-semantic-segmentation-on-xbd?p=changemamba-remote-sensing-change-detection)

</div>

## 🛎️更新日志
* **` 通知🐍🐍`**: ChangeMamba已经被[IEEE TGRS](https://ieeexplore.ieee.org/document/10565926)接收！仓库的代码已更新完毕！如果对您的研究有所帮助，请考虑给该仓库一个⭐️**star**⭐️！
* **` 2024年07月19日`**: ChangeMamba入选为[IEEE GRSS本周代表论文](https://www.linkedin.com/feed/update/urn:li:activity:7219970529498214400/)！！
* **` 2024年07月01日`**: 我们上传了经过预处理的[SECOND数据集](https://drive.google.com/file/d/1oApvN8RhJUjxO9pv6GikFbM0S4j8xXwE/view?usp=sharing)！！
* **` 2024年06月17日`**: ChangeMamba被[IEEE TGRS](https://ieeexplore.ieee.org/document/10565926)接收！
* **` 2024年06月08日`**: 中文版文档已上线！！
* **` 2024年04月18日`**: 我们已经发布了 ChangeMamba 模型在 BCD 任务中的所有权重。欢迎[使用](#%EF%B8%8F结果下载)！！
* **` 2024年04月05日`**: 该工作的[[ArXiv论文](https://arxiv.org/pdf/2404.03425.pdf)]已经上线！！
* **` 2024年04年05日`**: MambaBCD、MambaSCD 和 MambaBDA 的模型和训练代码已经整理并上传。欢迎使用！！

## 🔭简介

* [**ChangeMamba**](https://ieeexplore.ieee.org/document/10565926)系列模型包括三种有效的变化检测任务的基准模型，分别为二元变化检测模型MambaBCD、语义变化检测模型MambaSCD和建筑物损坏评估模型MambaBDA。

<p align="center">
  <img src="figures/network_architecture.png" alt="accuracy" width="90%">
</p>

* **ChangeMamba中的三种时空关系学习机制**

<p align="center">
  <img src="figures/STLM.png" alt="arch" width="60%">
</p>


## 🗝️开始使用
### `一、安装`
此仓库的代码是在 Linux 系统下运行的。我们尚未测试是否能在其他操作系统下运行。

首先需要安装[VMama仓库](https://github.com/MzeroMiko/VMamba)。以下安装顺序取自VMama仓库。


**步骤 1 —— 克隆仓库:**

克隆该版本库并导航至项目目录：
```bash
git clone https://github.com/ChenHongruixuan/MambaCD.git
cd MambaCD
```


**步骤 2 —— 环境设置:**

建议设置 conda 环境并通过 pip 安装依赖项。使用以下命令设置环境：

***创建并激活新的 conda 环境***

```bash
conda create -n changemamba
conda activate changemamba
```

***安装依赖项***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```


***检测和分割任务的依赖库（在 VMamba 中为可选项）***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```
### `二、下载预训练权重`
另外，请下载[VMamba-Tiny](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link), [VMamba-Small](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link), and [VMamba-Base](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link)在ImageNet上的预训练权重并把它们放在下述文件夹中 

```bash
project_path/MambaCD/pretrained_weight/
```

### `三、数据准备`
***二元变化检测***

论文使用了三个基准数据集 [SYSU](https://github.com/liumency/SYSU-CD)、[LEVIR-CD+](https://chenhao.in/LEVIR/) 和 [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) 用于评估模型的二元变化检测的性能。请下载这些数据集，并将其组织成下述文件夹/文件结构：

```
${DATASET_ROOT}   # 数据集根目录，例如: /home/username/data/SYSU
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
├── val
│   ├── ...
│   ...
│
├── test
│   ├── ...
│   ...
│ 
├── train.txt   # 数据名称列表，记录所有训练数据的名称
├── val.txt     # 数据名称列表，记录所有验证数据的名称
└── test.txt    # 数据名称列表，记录所有测试数据的名称
```

***语义变化检测***

语义变化检测任务的数据集为[SECOND数据集](https://captain-whu.github.io/SCD/)。 请下载该数据集，并使其具有以下文件夹/文件结构。请注意，**原始 SECOND 数据集中的土地覆盖图为 RGB 图像。您需要将其转换为单通道图像**。另外，**二元变化图需要您自行生成**，并将其放入文件夹 [`GT_CD`]。

或者，欢迎您直接下载并使用经过我们[预处理后的SECOND 数据集](https://drive.google.com/file/d/1oApvN8RhJUjxO9pv6GikFbM0S4j8xXwE/view?usp=sharing)。

```
${DATASET_ROOT}   # 数据集根目录，例如 /home/username/data/SECOND
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
│   ├── GT_CD   # 二元变化图
│   │   ├──00001.png 
│   │   ... 
│   │
│   ├── GT_T1   # T1时相的土地覆盖图
│   │   ├──00001.png 
│   │   ...  
│   │
│   └── GT_T2   # T2时相的土地覆盖图
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

***建筑物损坏评估***

xBD 数据集可从 [xView 2 挑战赛网站](https://xview2.org/dataset) 下载。下载后，请按以下结构进行组织： 
```
${DATASET_ROOT}   # 数据集根目录，例如：/home/username/data/xBD
├── train
│   ├── images
│   │   ├──guatemala-volcano_00000000_pre_disaster.png
│   │   ├──guatemala-volcano_00000000_post_disaster.png
│   │   ...
│   │
│   └── targets
│       ├──guatemala-volcano_00000003_pre_disaster_target.png
│       ├──guatemala-volcano_00000003_post_disaster_target.png
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
├── train.txt # 数据名称列表，记录所有训练数据的名称
├── test.txt  # 数据名称列表，记录所有测试数据的名称
└── holdout.txt  # 数据名称列表，记录所有留出集数据的名称
```


### `四、训练模型`
在训练模型之前，请进入 [`changedetection`]文件夹，其中包含网络定义、训练和测试的所有代码。

```bash
cd <project_path>/MambaCD/changedetection
```

***二元变化检测***

运行以下命令在 SYSU 数据集上训练和评估 MambaBCD-Small模型：
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

***语义变化检测***

运行以下命令在 SECOND 数据集上训练和评估 MambaSCD-Small模型：
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

运行以下命令在 xBD 数据集上训练和评估 MambaBDA-Small：
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
### `五、使用训练完成后的权重进行推理`

推理前，请先通过命令行进入 [`changedetection`]文件夹。
```bash
cd <project_path>/MambaCD/changedetection
```


***二元变化检测***

以下命令展示了如何在 LEVIR-CD+ 数据集上使用训练完成的 MambaBCD-Tiny 推断二元变化图：

* **`提示`**: 请使用 [--resume] 来加载训练过的模型，而不要使用 [--pretrained_weight_path]。 

```bash
python script/infer_MambaBCD.py  --dataset 'LEVIR-CD+' \
                                 --model_type 'MambaBCD_Tiny' \
                                 --test_dataset_path '<dataset_path>/LEVIR-CD+/test' \
                                 --test_data_list_path '<dataset_path>/LEVIR-CD+/test_list.txt' \
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'
                                 --resume '<saved_model_path>/MambaBCD_Tiny_LEVIRCD+_F1_0.8803.pth'
```

***语义变化检测***

以下命令展示了如何在 SECOND 数据集上使用训练完成的 MambaSCD-Tiny 推断语义变化图：
```bash
python script/infer_MambaBCD.py  --dataset 'SECOND'  \
                                 --model_type 'MambaSCD_Tiny' \
                                 --test_dataset_path '<dataset_path>/SECOND/test' \
                                 --test_data_list_path '<dataset_path>/SECOND/test_list.txt' \
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'
                                 --resume '<saved_model_path>/[your_trained_model].pth'
```

***建筑物损害评估***

以下命令展示了如何在 xBD 数据集上使用训练完成的 MambaBDA-Tiny 推断语义变化图：
```bash
python script/infer_MambaBDA.py  --dataset 'SECOND'  \
                                 --model_type 'MambaSCD_Tiny' \
                                 --test_dataset_path '<dataset_path>/xBD/test' \
                                 --test_data_list_path '<dataset_path>/xBD/test_list.txt' \
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'
                                 --resume '<saved_model_path>/[your_trained_model].pth'
```

## ⚗️结果下载


* *非常感谢您如果能给此 repo 一个⭐️**star**⭐️并且保持关注。*

* *请注意，由于我们上传到Github的代码是经过重新组织整理的。下面提供的模型权重也是采用重新组织整理后的代码训练得到的，因此精度可能会和原始论文不完全一致（大多数情况都高于论文中报告的精度）。*

### `一、VMamba（编码器）的预训练权重`

| 方法 | ImageNet (ckpt) | 
| :---: | :---: |
| VMamba-Tiny | [[GDrive](https://drive.google.com/file/d/160PXughGMNZ1GyByspLFS68sfUdrQE2N/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1P9KRVy4lW8LaKJ898eQ_0w?pwd=7qxh)] |   
| VMamba-Small | [[GDrive](https://drive.google.com/file/d/1dxHtFEgeJ9KL5WiLlvQOZK5jSEEd2Nmz/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1RRjTA9ONhO43sBLp_a2TSw?pwd=6qk1)]   | 
| VMamba-Base |  [[GDrive](https://drive.google.com/file/d/1kUHSBDoFvFG58EmwWurdSVZd8gyKWYfr/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/14_syzqwNnVB8rD3tejEZ4w?pwd=q825)] | 

### `二、二元变化检测`

| 方法 | SYSU (ckpt) | LEVIR-CD+ (ckpt) | WHU-CD (ckpt) | 
| :---: | :---: | :---: | :---: |
| MambaBCD-Tiny | [[GDrive](https://drive.google.com/file/d/1qoivh0zrZjpPzUOiIxLWZn7kdBQ-MqnY/view?usp=sharing)][[BaiduYun](https://pan.baidu.com/s/160RiqDQKB6rBwn7Fke6xFQ?pwd=wqf9)] |  [[GDrive](https://drive.google.com/file/d/1AtiXBBCoofi1e5g4STYUzBgJ1fYN4VhN/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/13dGC_J-wyIfoPwoPJ5Uc6Q?pwd=8ali)]	 | [[GDrive](https://drive.google.com/file/d/1ZLKXhGKgnWoyS0X8g3HS45a3X1MP_QE6/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1DhTedGZdIC80y06tog1xbg?pwd=raf0)] | 
| MambaBCD-Small | [[GDrive](https://drive.google.com/file/d/1ZEPF6CvvFynL-yu_wpEYdpHMHl7tahpH/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1f8iwuKCkElU9rc24_ZzXBw?pwd=46p5)]   | [[GDrive](https://drive.google.com/file/d/19jEBLheCwEnQqF23EqNrn1r79D-nZ95y/view?usp=sharing)][[BaiduYun](https://pan.baidu.com/s/1EKWp-tF0EEGgZ-nVlW8S1g?pwd=n3qz)]  | [[GDrive](https://drive.google.com/file/d/1ejiBIhSAJF0P65Xn6DpzRpARiIGPLiWw/view?usp=drive_link)][[BaiduYun]](https://pan.baidu.com/s/1tIWyfJa2o9EMwrKg-gKTnw?pwd=vizm) | 
| MambaBCD-Base |  [[GDrive](https://drive.google.com/file/d/14WbK9KjOIOWuea3JAgvIfyDvqACExZ0s/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1xiWWjlhuJWA40cMggevdlA?pwd=4jft)] | [[GDrive](https://drive.google.com/file/d/1uQy5tGXW20xFZvF7hIvZvsi7-JU7tg7G/view?usp=drive_link)] [[BaiduYun](https://pan.baidu.com/s/1M_u7HdIEFIEA2d3L1kfu3Q?pwd=rkgp)] | [[GDrive](https://drive.google.com/file/d/1K7aSuT3os7LR9rUvoyVNP-x0hWKZocrn/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1o6Z6ecIJ59K9eB2KqNMD9w?pwd=4mqd)] |

### `三、语义变化检测`
| 方法 |  SECOND (ckpt) | SECOND (results) |
| :---: | :---: | :---: | 
| MambaSCD-Tiny |  --  |  --  | 
| MambaSCD-Small | --  | -- | 
| MambaSCD-Base | --  |  --  | 



### `四、建筑物损害评估`
| 方法 |  xBD (ckpt) | xBD (results) |
| :---: | :---: | :---: | 
| MambaBDA-Tiny |  -- | --  | 
| MambaBDA-Small | -- | -- |
| MambaBDA-Base | -- | -- | 

## 🤔常见问题
下面列出了一些常见问题的快速、简便的解决方案。

| 问题 | 解决方案 | 
| :---: | :---: | 
| 关于SECOND数据集的问题 | 请参考 Issue [#13](https://github.com/ChenHongruixuan/MambaCD/issues/13) / [#22](https://github.com/ChenHongruixuan/MambaCD/issues/22) / [#45](https://github.com/ChenHongruixuan/MambaCD/issues/45) |
| CUDA out of memory issue | 请降低训练和评估的batch size  |
| 修改模型结构 | 请参考 Issue [#44](https://github.com/ChenHongruixuan/MambaCD/issues/44)  |
| 关于iteration、epoch和batch size之间的关系 | 请参考 Issue [#32](https://github.com/ChenHongruixuan/MambaCD/issues/32) / [#48](https://github.com/ChenHongruixuan/MambaCD/issues/48)  |
NameError: name 'selective_scan_cuda_oflex' is not defined | 请参考 Issue [#9](https://github.com/ChenHongruixuan/MambaCD/issues/9)


## 📜引用

如果我们的代码有助于您的研究，请考虑引用我们的论文，并给我们一个⭐️star⭐️ :)
```
@article{chen2024changemamba,
  author={Hongruixuan Chen and Jian Song and Chengxi Han and Junshi Xia and Naoto Yokoya},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model}, 
  year={2024},
  volume={62},
  number={},
  pages={1-20},
  doi={10.1109/TGRS.2024.3417253}
}
```


## 🤝致谢
本项目采用和借鉴了([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ScanNet ([paper](https://arxiv.org/abs/2212.05245), [code](https://github.com/ggsDing/SCanNet)), BDANet ([paper](https://ieeexplore.ieee.org/document/9442902), [code](https://github.com/ShaneShen/BDANet-Building-Damage-Assessment))等仓库。感谢他们的优秀工作！

## 🙋联系我们
***如有任何问题，请随时[联系我们。](mailto:Qschrx@gmail.com)***
