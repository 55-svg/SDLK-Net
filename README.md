# SDLK-Net: Salient Object Detection Network

## Overview
SDLK-Net is an advanced network for RGB-D and RGB-T salient object detection, featuring three key modules:
- Squeezed Large Kernel Edge-Enhanced Fusion (SLKEF)
- Adaptive Extraction Multi-path Matching (AEMM)
- Multi-Scale Filter Gate Integration (MSFGI)

## Requirements
- Python ≥ 3.8
- CUDA 11
- PyTorch

## Installation
1. Create conda environment:
```bash
conda create -n sdlknet python=3.9
conda activate sdlknet
```

2. Clone repository:
```bash
git clone https://github.com/[your_username]/SDLK-Net.git
cd SDLK-Net
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python train.py --dataset rgbd  # for RGB-D dataset
python train.py --dataset rgbt  # for RGB-T dataset
```

### Evaluation
```bash
python evaluate.py --dataset rgbd --model_path [model_weights.pth]
```

## Results
Pre-trained models and saliency maps available upon request.
