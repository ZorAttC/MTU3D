# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

这是 MTU3D (Move to Understand a 3D Scene) 的官方实现，这是一个统一的框架，将主动感知与3D视觉语言学习相结合，用于具身导航。该代码库支持在多个具身导航基准测试上进行训练和评估，包括 HM3D-OVON、GOAT-Bench、SG3D 和 A-EQA。

## 核心架构组件

### 训练流程
- **阶段1**: 低级感知训练，用于具身分割 (`embodied_scan_instseg.yaml`)
- **阶段2**: 视觉语言探索预训练 (`embodied_vle.yaml`)
- **阶段3**: 导航数据集特定的微调

### 核心模型
- **EmbodiedPQ3DInstSegModel**: 用于3D实例分割和视觉定位的主模型
- **Query3DVLE**: 用于具身导航任务的视觉语言编码器
- **PCDMask3DSegLevelEncoder**: 使用 MinkowskiEngine 的点云编码器
- **ObjectEncoder**: 多视角图像特征编码器

### 关键依赖
- **MinkowskiEngine**: 用于点云处理的稀疏3D卷积引擎
- **Habitat-Lab**: 用于仿真和评估的具身AI框架
- **FastSAM**: 用于2D感知的快速分割模型
- **Hydra**: 配置管理系统

## 常用命令

### 环境设置与安装
```bash
# 创建环境
conda env create -n mtu3d python=3.8
pip install torch==2.0.0 torchvision==0.15.1
pip install -r requirements.txt

# 安装 MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# 安装 Habitat
conda install habitat-sim=0.2.3 headless -c conda-forge -c aihabitat
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

### 训练命令
```bash
# 阶段1: 感知训练
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_scan_instseg.yaml

# 阶段2: VLE预训练
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_vle.yaml

# 多GPU训练
python launch.py --mode submitit --partition gpu --gpu_per_node=4 --config configs/embodied-pq3d-final/embodied_vle.yaml

# 调试模式
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_scan_instseg.yaml debug.flag=True debug.debug_size=10
```

### 评估命令
```bash
# 设置环境
export PYTHONPATH=./:./hm3d-online:./hm3d-online/FastSAM
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False

# 运行评估（先编辑相应文件中的路径）
bash run_nav.sh  # 默认使用 goat-nav.py
```

### 配置文件
- **训练配置**: `configs/embodied-pq3d-final/`
  - `embodied_scan_instseg.yaml`: 阶段1感知训练
  - `embodied_vle.yaml`: 阶段2 VLE预训练
- **数据配置**: 在配置文件中修改路径以适配您的数据位置
- **导航配置**: `hm3d-online/*-nav.py` 文件用于评估

### 数据要求
- **SceneVerse**: 基础3D场景数据（设置 `data.scene_verse_base`）
- **EmbodiedScan**: 阶段1的训练数据（设置 `data.embodied_base`）
- **VLE数据**: 视觉语言探索数据（设置 `data.embodied_vle`）
- **HM3D**: Habitat仿真环境（在 `hm3d-online/*-nav.py` 中设置）

### 关键入口点
- `run.py`: 带有Hydra配置的主训练脚本
- `launch.py`: 支持SLURM的多GPU训练启动器
- `hm3d-online/goat-nav.py`: GOAT-bench评估
- `hm3d-online/hm3d-nav.py`: HM3D-OVON评估
- `hm3d-online/sg3d-nav.py`: SG3D评估