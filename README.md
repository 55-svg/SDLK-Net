SDLK-Net: Enhanced Squeezed Directional Large Kernel Network for Salient Object Detection
Official implementation of the paper:
"SDLK-Net: Enhanced Squeezed Directional Large Kernel Multi-Scale Multi-modal Fusion Network for Salient Object Detection"
Lingyu Yan, Ting Zhou, Rong Gao, Zengmao Wang, Zhiwei Ye, Xinyun Wu

Overview
A novel network for RGB-D and RGB-T salient object detection, featuring:

Squeezed Large Kernel Edge-Enhanced Fusion (SLKEF)

Adaptive Extraction Multi-path Matching (AEMM)

Multi-Scale Filter Gate Integration (MSFGI)

Quick Start
Requirements
Python ≥ 3.8

CUDA 11

PyTorch

Installation
bash
conda create -n sdlknet python=3.9
conda activate sdlknet
git clone [repository_url]
cd SDLK-Net
Training
bash
python train.py
Results
Pre-computed saliency maps and evaluation code available upon request from the corresponding author.
