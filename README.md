# Convolution-based mesh denoising under various noise types

Our project are based on the papers: [Learning Self-prior for Mesh Denoising using Dual Graph Convolutional Networks (2022)](https://github.com/astaka-pe/Dual-DMP) and [Deep Image Prior (2020)](https://arxiv.org/abs/1711.10925) and [Early Stopping for Deep Image Prior (2023)](https://arxiv.org/abs/2112.06074)

This project focuses on mesh denoising under various noise types using an unsupervised deep-learning method. The approach is based on two graph convolutional networks—PosNet and NormNet—that are trained jointly to filter vertex positions and facet normals separately. A key aspect of this work is addressing the challenge of early stopping, improving training stability to ensure robust and efficient mesh smoothing.

![anim](https://github.com/user-attachments/assets/7c466a77-0337-4458-aa5c-945cfa23a00c)

**Source:** Hattori, S., Yatagawa, T., Ohtake, Y., & Suzuki, H. (2022). *Learning self-prior for mesh denoising using dual graph convolutional networks*. In *Proceedings of the European Conference on Computer Vision (ECCV)*. [https://doi.org/10.1007/978-3-031-20062-5_21](https://doi.org/10.1007/978-3-031-20062-5_21)

## Method Overview

![diagram](https://github.com/user-attachments/assets/b791550a-b563-423d-bdfa-127809c3d913)

## Results

![reconstructions png](https://github.com/user-attachments/assets/b2d77bc5-7390-4989-9862-713fdd3db869)

___

## Getting Started

### 0. Environments

<img src="https://img.shields.io/badge/GPU-NVIDIA_GeForce_RTX_4070_12GB-blue" alt="NVIDIA GeForce TITAN X 12GB">

```
python==3.10
torch==1.13.1
torch-geometric==2.2.0
```

### 1. Installation
```
git clone -b g6-main https://github.com/tub-cv-group/htcv_ss2425_dlfor3d.git
cd htcv_ss2425_dlfor3d/Code
docker image build -t astaka-pe/ddmp .
docker run -itd --gpus all -p 8080:8080 --name ddmp -v .:/work astaka-pe/ddmp
docker exec -it ddmp /bin/bash
```
<!-- conda env create -f environment.yml
conda activate ddmp -->

### 2. Preparation

The Dataset is distributed as a zip file. Please unzip and place it under Code. 

```
unzip datasets.zip
```
### 3. Preprocessing
Place a mesh under `datasets/{model-name}/` as an obj file.

Run this if you have real scan data:
this will generate a noisy and smoothed version of your model (the noisy version is the original scan that already contains noise)
```
python3 preprocess/preprocess.py -i datasets/{model-name}
```

Run this instead if you have synthetic data:
This will generate different types of noise (gaussian, poisson, mix of gaussian and poisson, salt & pepper, salt & pepper surface, sinus and sinus space) for your model with a ground truth model and a smoothed version
```
python3 preprocess/noisemaker.py -i datasets/{model-name}/{model-name}.obj
```

### 4. Training
Run this if you want to visualize your training with early stopping on a localhost server:
```
python3 main.py -i datasets/{model-name} -es True
```

Run this instead if you want to train with early stopping without visualiziation:
```
python3 main4real.py -i datasets/{model-name} -es True
```

Depending on whether it is a CAD or non-CAD model, the parameters and weights must be set differently.
You should set appropriate weights as described in this [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630358.pdf)

- CAD model
```
python3 main.py -i datasets/fandisk --k1 3 --k2 0 --k3 3 --k4 4 --k5 2 --bnfloop 5 -es True
```

- Non-CAD model
```
python3 main.py -i datasets/ankylosaurus --k1 3 --k2 4 --k3 4 --k4 4 --k5 1 --bnfloop 1 -es True
```

- Real-scanned model
```
python3 main.py -i datasets/pyramid -es True
```

You can monitor the training progress through the web viewer. (Default: http://localhost:8080)

![flamingo_first_gaussian_reconstruction](https://github.com/user-attachments/assets/c744a6e6-61cf-46f9-8a21-c474fe027725)

Outputs will be generated under `datasets/{model-name}/output/` with their MAD scores (if there is a ground truth model, else the MAD Score will be displayed with the value 0.0).


### Visualization
If you want to visualize the angular difference between the face normals of the ground truth and its reconstruction then you run the viz_diffs.py in the visualize folder. It is not part of the docker container.
```
python viz_diffs.py -i datasets/{model-name}
```
![80d364f0-cb21-4f6d-8558-ab65e02e66b4_000](https://github.com/user-attachments/assets/910c88e1-61ed-4a27-99ee-12ea5d21510e)


In the visualize folder are also other visualization tools to generate metrics.
