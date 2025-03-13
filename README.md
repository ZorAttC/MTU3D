## Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation

<p align="left">
    <a href='https://arxiv.org/abs/2405.11442'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2405.11442'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://pq3d.github.io'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <!-- <a href='https://huggingface.co/spaces/li-qing/PQ3D-Demo'> -->
      <!-- <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggigFace'> -->
    <!-- </a> -->
    <a href='https://drive.google.com/drive/folders/1MDt9yaco_TllGfqqt76UOxMV1JMMVsji?usp=share_link'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints(TODO)'>
    </a>
</p>

[Ziyu Zhu](https://zhuziyu-edward.github.io/), 
[Xilin Wang](),
[Yixuan Li](),
[Zhuofan Zhang](https://tongclass.ac.cn/author/zhuofan-zhang/), 
[Xiaojian Ma](https://jeasinema.github.io/), 
[Yixin Chen](https://yixchen.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io),
[Wei Liang](),
[Qian Yu](),
[Zhidong Deng](https://www.cs.tsinghua.edu.cn/csen/info/1165/4052.htm)📧,
[Siyuan Huang](https://siyuanhuang.com/)📧,
[Qing Li](https://liqing-ustc.github.io/)📧

This repository is the official implementation of the Arxiv paper "Move to Understand a 3D Scene: Briding Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation".

[Paper](https://arxiv.org/abs/2405.11442) |
[arXiv](https://arxiv.org/abs/2405.11442) |
[Project](https://pq3d.github.io) |
[Checkpoints](https://drive.google.com/drive/folders/1MDt9yaco_TllGfqqt76UOxMV1JMMVsji?usp=share_link)

<div align=center>
<img src='https://mtu3d.github.io/mtu3d-teaser.png' width=100%>
</div>

### News
- [ 2024.08 ] Release training and evaluation.
<!-- - [ 2024.07 ] Our huggingface DEMO is here [DEMO](https://huggingface.co/spaces/li-qing/PQ3D-Demo), welcome to try our model!
- [ 2024.07 ] Release codes of model! TODO: Clean up training and evaluation -->

### Abstract

 Embodied scene understanding requires not only comprehending visual-spatial information that has been observed but also determining where to explore next in the 3D physical world. 
  Existing 3D Vision-Language (3D-VL) models primarily focus on grounding objects in static observations from 3D reconstruction, such as meshes and point clouds, but lack the ability to actively perceive and explore their environment. To address this limitation, we introduce **M**ove **t**o **U**nderstand (**MTU3D**), a unified framework that integrates active perception with 3D vision-language learning, enabling embodied agents to effectively explore and understand their environment. . Extensive evaluations across various embodied navigation and question-answering benchmarks show that MTU3D outperforms state-of-the-art reinforcement learning and modular navigation approaches by 14\%, 27\%, 11\%, and 3\% in success rate on HM3D-OVON, GOAT-Bench, SG3D, and A-EQA, respectively. MTU3D's versatility enables navigation using diverse input modalities, including categories, language descriptions, and reference images. The deployment on a real robot demonstrates MTU3D's effectiveness in handling real-world data. These findings highlight the importance of bridging visual grounding and exploration for embodied intelligence.

### Install
1. Install conda package
```
conda env create --name envname 
pip3 install torch==2.0.0
pip3 install torchvision==0.15.1
pip3 install -r requirements.txt
```

2. Install Minkowski Engine
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
conda install openblas-devel -c anaconda
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

3. Install FastSAM, link is here [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM.git)
```
cd hm3d-online
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt
cd ../..
```

4. Install HabitatSim and HabitatLab
```
conda install habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y
git clone --branch v0.2.3 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

### Prepare data
1. download sceneverse data from [scene_verse_base](https://github.com/scene-verse/sceneverse?tab=readme-ov-file) and change `data.scene_verse_base` to sceneverse data directory.
2. download stage1 data for embodied segmentation training from [stage1]() and change `data.embodied_base` to download data directory.
3. download feature saved from stage1 from  [stage1_feat](https://drive.google.com/drive/folders/12BjbhXzV7lON4X0tx3e7DdvZHhI1Q1f2?usp=share_link) and change `data.embodied_feat` to download data directory.
4. download vle data from [vle_stage2]() and change `data.embodied_vle` to download data directory
5. download hm3d data from [hm3d]() and change `hm3d_data_base_path` in hm3d-online/*.-nav.py
5. download embodied navigation benchmark data from [embodied-bench]() and change `data_set_path` and `navigation_data_path` in hm3d-online/*.nav.py


### Prepare checkpoints
1. download [mtu3d-stage1](), [mtu3d-stage2](), and change `pq3d_stage1_path` and `pq3d_stage2_path` in hm3d-online/*-nav.py


### Run MTU3D for training
Stage 1 low-level percetpion training
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_scan_instseg.yaml
```
Stage 2 vision-langauge-exploration pre-training
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_vle.yaml 
```
Stage 3 navigation dataset specific fine-tuning
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_vle.yaml data.train=[{specific_dataset}] pretrain_ckpt_path={stage2_pretrained_path}
```
For multi-gpu training usage, we use four GPU in our experiments.
```
python launch.py --mode ${launch_mode} \
    --qos=${qos} --partition=${partition} --gpu_per_node=4 --port=29512 --mem_per_gpu=80 \
    --config {config}  \
```

### Run MTU3D for evaluation
Evaluation for HM3D-ovon
```
edit run_nav.sh and use "python3 hm3d-online/ovon-nav.py"
bash run_nav.sh
```
Evaluation for Goat-bench
```
edit run_nav.sh and use "python3 hm3d-online/goat-nav.py"
bash run_nav.sh
```
Evaluation for SG3D
```
edit run_nav.sh and use "python3 hm3d-online/sg3d-nav.py"
bash run_nav.sh
```

### Acknowledgement
We would like to thank the authors of [Vil3dref](https://github.com/cshizhe/vil3dref), [Mask3d](https://github.com/JonasSchult/Mask3D), [Openscene](https://github.com/pengsongyou/openscene), [Xdecoder](https://github.com/microsoft/X-Decoder), and [3D-VisTA](https://github.com/3d-vista/3D-VisTA) for their open-source release.


### Citation:
```
@article{zhu2024unifying,
    title={Unifying 3D Vision-Language Understanding via Promptable Queries},
    author={Zhu, Ziyu and Zhang, Zhuofan and Ma, Xiaojian and Niu, Xuesong and Chen, Yixin and Jia, Baoxiong and Deng, Zhidong and Huang, Siyuan and Li, Qing},
    journal={arXiv preprint arXiv:2405.11442},
    year={2024}
}
```
