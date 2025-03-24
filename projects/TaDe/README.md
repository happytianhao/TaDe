conda create -n mmdet3d python=3.8.5 -y
conda activate mmdet3d
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine==0.10.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet==3.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
# "-b dev-1.x" 表示切换到 `dev-1.x` 分支。
cd mmdetection3d
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
将mmdet和mmdet3d的<改成<=

ln -s /path/to/nuScenes data/nuscenes

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


python projects/TaDe/tools/create_bev_map_nuscenes.py 
python projects/TaDe/tools/prepare_data_nuscenes.py

python tools/train.py projects/TaDe/configs/end2end.py
