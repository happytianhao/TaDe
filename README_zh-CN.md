# 通过任务分解改进鸟瞰图语义分割

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2404.01925-b31b1b.svg)](https://arxiv.org/abs/2404.01925)
[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-blue.svg)](https://cvpr.thecvf.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[English](README.md) | 简体中文

</div>

## 简介

本仓库包含论文 **Improving Bird's Eye View Semantic Segmentation by Task Decomposition** 的官方实现代码，该论文发表于 **CVPR 2024**。

**作者：** Tianhao Zhao, Yongcan Chen, Yu Wu, Tianyang Liu, Bo Du, Peilun Xiao, Shi Qiu, Hongda Yang, Guozhen Li, Yi Yang, Yutian Lin

**摘要：** 鸟瞰图（BEV）语义分割在自动驾驶中起着至关重要的作用。以往的方法通常遵循端到端的流程，直接从单目RGB输入预测BEV分割图。然而，当RGB输入和BEV目标来自不同视角时，会带来挑战，使得直接的点对点预测难以优化。在本文中，我们将原始的BEV分割任务分解为两个阶段，即BEV地图重建和RGB-BEV特征对齐。在第一阶段，我们训练一个BEV自编码器，在给定损坏的噪声潜在表示的情况下重建BEV分割图，这促使解码器学习典型BEV模式的基础知识。第二阶段涉及将RGB输入图像映射到第一阶段的BEV潜在空间，在特征层面直接优化两个视图之间的相关性。我们的方法将感知和生成的复杂性简化为不同的步骤，使模型能够有效地处理复杂和具有挑战性的场景。此外，我们提出将BEV分割图从笛卡尔坐标系转换为极坐标系，以建立RGB图像和BEV地图之间的列对应关系。而且，我们的方法既不需要多尺度特征，也不需要相机内参来进行深度估计，节省了计算开销。在nuScenes和Argoverse上的大量实验表明了我们方法的有效性和效率。

## 新闻

- **[2024/08]** 代码和模型权重发布。
- **[2024/02]** 论文被CVPR 2024接收。

## 特性

- ✅ BEV语义分割的任务分解方法
- ✅ BEV自编码器学习基础BEV模式
- ✅ 潜在空间中的RGB-BEV特征对齐
- ✅ 极坐标转换以获得更好的RGB-BEV对应关系
- ✅ 无需多尺度特征或相机内参
- ✅ 支持nuScenes和Argoverse数据集
- ✅ 三阶段训练流程：自编码器 → 对齐 → 微调

## 安装

### 环境要求
- Python 3.8.5
- PyTorch
- MMEngine 0.10.7
- MMCV 2.2.0
- MMDetection 3.3.0
- MMDetection3D (dev-1.x分支)

### 环境配置

```bash
# 创建conda环境
conda create -n mmdet3d python=3.8.5 -y
conda activate mmdet3d

# 安装PyTorch
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装OpenMMLab相关包
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine==0.10.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet==3.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆并安装MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装
```

### 克隆本仓库

```bash
git clone https://github.com/happytianhao/TaDe.git
cd TaDe
```

## 数据准备

### nuScenes数据集

1. 下载 [nuScenes数据集](https://www.nuscenes.org/download) 并按如下方式组织：

```
data/nuscenes
├── maps
│   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
│   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
│   ├── 53992ee3023e5494b90c316c183be829.png
│   ├── 93406b464a165eaba6d9de76ca09f5da.png
│   ├── basemap
│   ├── expansion
│   └── prediction
├── samples
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   └── LIDAR_TOP
├── sweeps
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   └── LIDAR_TOP
└── v1.0-trainval
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── sample.json
    ├── scene.json
    ├── sensor.json
    └── visibility.json
```

2. 创建软链接：

```bash
ln -s /path/to/nuScenes data/nuscenes
```

3. 准备BEV地图和处理后的数据：

```bash
python projects/TaDe/tools/create_bev_map_nuscenes.py
python projects/TaDe/tools/prepare_data_nuscenes.py
```

## 训练

### 端到端训练

```bash
python tools/train.py projects/TaDe/configs/end2end.py
```

### 三阶段训练（推荐）

**阶段1：自编码器预训练**
```bash
python tools/train.py projects/TaDe/configs/tade_stage1_autoencoder.py
```

**阶段2：任务对齐**
```bash
python tools/train.py projects/TaDe/configs/tade_stage2_alignment.py
```

**阶段3：微调**
```bash
python tools/train.py projects/TaDe/configs/tade_stage3_finetuning.py
```

## 测试

```bash
python tools/test.py projects/TaDe/configs/end2end.py [checkpoint_path] --eval [metrics]
```

## 模型库

| 模型 | 训练策略 | mAP | NDS | 配置文件 | 权重 |
|------|---------|-----|-----|---------|------|
| TaDe | 端到端  | -   | -   | [config](projects/TaDe/configs/end2end.py) | [model]() |
| TaDe | 三阶段  | -   | -   | [config](projects/TaDe/configs/tade_stage3_finetuning.py) | [model]() |

## 引用

如果您在研究中使用了本工作，请考虑引用：

```bibtex
@inproceedings{zhao2024improving,
  title={Improving Bird's Eye View Semantic Segmentation by Task Decomposition},
  author={Zhao, Tianhao and Chen, Yongcan and Wu, Yu and Liu, Tianyang and Du, Bo and Xiao, Peilun and Qiu, Shi and Yang, Hongda and Li, Guozhen and Yang, Yi and Lin, Yutian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## 许可证

本项目基于 [Apache 2.0 license](LICENSE) 开源。

## 致谢

本代码基于 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) 构建。感谢作者们的出色工作和开源贡献。

## 联系方式

如有问题或讨论，请提交issue或联系 zthwhucs@gmail.com。

---

**注意：** 原始的MMDetection3D README可以在 [README_mmdet3d_zh-CN.md](README_mmdet3d_zh-CN.md) 中找到。
