# Improving Bird's Eye View Semantic Segmentation by Task Decomposition

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2404.01925-b31b1b.svg)](https://arxiv.org/abs/2404.01925)
[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-blue.svg)](https://cvpr.thecvf.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## Introduction

This repository contains the official implementation of **Improving Bird's Eye View Semantic Segmentation by Task Decomposition**, published at **CVPR 2024**.

**Authors:** Tianhao Zhao, Yongcan Chen, Yu Wu, Tianyang Liu, Bo Du, Peilun Xiao, Shi Qiu, Hongda Yang, Guozhen Li, Yi Yang, Yutian Lin

**Abstract:** Semantic segmentation in bird's eye view (BEV) plays a crucial role in autonomous driving. Previous methods usually follow an end-to-end pipeline, directly predicting the BEV segmentation map from monocular RGB inputs. However, the challenge arises when the RGB inputs and BEV targets from distinct perspectives, making the direct point-to-point predicting hard to optimize. In this paper, we decompose the original BEV segmentation task into two stages, namely BEV map reconstruction and RGB-BEV feature alignment. In the first stage, we train a BEV autoencoder to reconstruct the BEV segmentation maps given corrupted noisy latent representation, which urges the decoder to learn fundamental knowledge of typical BEV patterns. The second stage involves mapping RGB input images into the BEV latent space of the first stage, directly optimizing the correlations between the two views at the feature level. Our approach simplifies the complexity of combining perception and generation into distinct steps, equipping the model to handle intricate and challenging scenes effectively. Besides, we propose to transform the BEV segmentation map from the Cartesian to the polar coordinate system to establish the column-wise correspondence between RGB images and BEV maps. Moreover, our method requires neither multi-scale features nor camera intrinsic parameters for depth estimation and saves computational overhead. Extensive experiments on nuScenes and Argoverse show the effectiveness and efficiency of our method.

## News

- **[2024/08]** Code and checkpoints released.
- **[2024/02]** Paper accepted to CVPR 2024.

## Features

- ✅ Task decomposition approach for BEV semantic segmentation
- ✅ BEV autoencoder for learning fundamental BEV patterns
- ✅ RGB-BEV feature alignment in latent space
- ✅ Polar coordinate transformation for better RGB-BEV correspondence
- ✅ No requirement for multi-scale features or camera intrinsic parameters
- ✅ Support for nuScenes and Argoverse datasets
- ✅ Three-stage training pipeline: autoencoder → alignment → finetuning

## Installation

### Prerequisites
- Python 3.8.5
- PyTorch
- MMEngine 0.10.7
- MMCV 2.2.0
- MMDetection 3.3.0
- MMDetection3D (dev-1.x branch)

### Environment Setup

```bash
# Create conda environment
conda create -n mmdet3d python=3.8.5 -y
conda activate mmdet3d

# Install PyTorch
pip install torch torchvision

# Install OpenMMLab packages
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0
mim install mmdet==3.3.0

# Clone and install MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .
```

### Clone this Repository

```bash
git clone https://github.com/happytianhao/TaDe.git
cd TaDe
```

## Data Preparation

### nuScenes Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download) and organize it as follows:

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

2. Create symbolic link:

```bash
ln -s /path/to/nuScenes data/nuscenes
```

3. Prepare BEV maps and processed data:

```bash
python projects/TaDe/tools/create_bev_map_nuscenes.py
python projects/TaDe/tools/prepare_data_nuscenes.py
```

## Training

### End-to-End Training

```bash
python tools/train.py projects/TaDe/configs/end2end.py
```

### Three-Stage Training (Recommended)

**Stage 1: Autoencoder Pre-training**
```bash
python tools/train.py projects/TaDe/configs/tade_stage1_autoencoder.py
```

**Stage 2: Task Alignment**
```bash
python tools/train.py projects/TaDe/configs/tade_stage2_alignment.py
```

**Stage 3: Fine-tuning**
```bash
python tools/train.py projects/TaDe/configs/tade_stage3_finetuning.py
```

## Testing

```bash
python tools/test.py projects/TaDe/configs/end2end.py [checkpoint_path] --eval [metrics]
```

## Model Zoo

| Model | Training Strategy | mAP | NDS | Config | Checkpoint |
|-------|------------------|-----|-----|--------|------------|
| TaDe  | End-to-End       | -   | -   | [config](projects/TaDe/configs/end2end.py) | [model]() |
| TaDe  | Three-Stage      | -   | -   | [config](projects/TaDe/configs/tade_stage3_finetuning.py) | [model]() |

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{zhao2024improving,
  title={Improving Bird's Eye View Semantic Segmentation by Task Decomposition},
  author={Zhao, Tianhao and Chen, Yongcan and Wu, Yu and Liu, Tianyang and Du, Bo and Xiao, Peilun and Qiu, Shi and Yang, Hongda and Li, Guozhen and Yang, Yi and Lin, Yutian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is built upon [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). We thank the authors for their excellent work and open-source contribution.

## Contact

For questions and discussions, please open an issue or contact zthwhucs@gmail.com.

---

**Note:** The original MMDetection3D README can be found in [README_mmdet3d.md](README_mmdet3d.md).
