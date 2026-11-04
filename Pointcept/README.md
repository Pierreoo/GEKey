<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo.png" width="400">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

[![Formatter](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml/badge.svg)](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml)

**Pointcept** is a powerful and flexible codebase for point cloud perception research.
## Citation
If you find _Pointcept_ useful to your research, please cite our work as encouragement. (੭ˊ꒳​ˋ)੭✧
```
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## Overview

- [Installation](#installation)
- [Acknowledgement](#acknowledgement)

## Installation

**GEKey:** after setting up the environment with one of the methods below, install the additional dependencies:
```bash
pip install geomloss SharedArray scikit-learn natsort trimesh seaborn open3d
```

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above (NVIDIA GPUs), or ROCm 7.0 and above (AMD GPUs).
- PyTorch: 1.10.0 and above.

### Conda Environment
- **Method 1**: Utilize conda `environment.yml` to create a new environment with one line code:
  ```bash
  # Create and activate conda environment named as 'pointcept-torch2.5.0-cu12.4'
  # cuda: 12.4, pytorch: 2.5.0

  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate pointcept-torch2.5.0-cu12.4
  ```

- **Method 2**: Use our pre-built Docker image and refer to the supported tags [here](https://hub.docker.com/r/pointcept/pointcept/tags). Quickly verify the Docker image on your local machine with the following command:
  ```bash
  docker run --gpus all -it --rm pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel bash
  git clone https://github.com/facebookresearch/sonata
  cd sonata
  export PYTHONPATH=./ && python demo/0_pca.py
  # Ignore the GUI error, we cannot expect a container to have its GUI, right?
  ```

- **Method 3**: Manually create a conda environment:
  ```bash
  conda create -n pointcept python=3.10 -y
  conda activate pointcept
  
  # (Optional) If no CUDA installed
  conda install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y
  
  conda install ninja -y
  # Choose version you want here: https://pytorch.org/get-started/previous-versions/
  conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y
  conda install h5py pyyaml -c anaconda -y
  conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
  conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
  pip install torch-geometric

  # spconv (SparseUNet)
  # refer https://github.com/traveller59/spconv
  pip install spconv-cu124

  # PPT (clip)
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git

  # transformers and peft
  pip install transformers==4.50.3
  pip install peft

  # PTv1 & PTv2 or precise eval
  cd libs/pointops
  # usual
  python setup.py install
  # docker & multi GPU arch
  TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
  # e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
  TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
  # on AMD GPUs (ROCm), select the target arch with PYTORCH_ROCM_ARCH instead
  # e.g. gfx90a: MI200; gfx1100: RX 7900 More available in: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
  PYTORCH_ROCM_ARCH="gfx90a;gfx1100" python setup.py install
  cd ../..

  # Open3D (visualization, optional)
  pip install open3d
  ```

## Acknowledgement
_Pointcept_ is designed by [Xiaoyang](https://xywu.me/), named by [Yixing](https://github.com/yxlao) and the logo is created by [Yuechen](https://julianjuaner.github.io/). It is derived from [Hengshuang](https://hszhao.github.io/)'s [Semseg](https://github.com/hszhao/semseg) and inspirited by several repos, e.g., [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
