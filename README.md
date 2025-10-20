# StyleDiffusion Reproduction

Recreation of the Style Diffusion research paper from ICCV 2023. [Paper link](https://arxiv.org/abs/2308.07863). 

### Setup Instructions

1. Clone the repository

```
git clone https://github.com/Ardacandra/style_diffusion_reproduction.git
cd style_diffusion_reproduction
```

2. Create environment

```
conda create -n style_diffusion_reproduction python=3.12
conda activate style_diffusion_reproduction
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.
pip install git+https://github.com/openai/CLIP.git
```