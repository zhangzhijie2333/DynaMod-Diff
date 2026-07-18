# DynaMod-Diff

Official implementation of **DefDynaRoute-Diff**, a diffusion-based framework for controllable industrial anomaly image generation. The method introduces Spatially Conditioned Dynamic Parameter Routing into the denoising U-Net and supports mask-guided defect synthesis while preserving the normal object appearance and background texture.
<p align="center">
  <img src="assets/result_display_figure.png" width="800">
</p>
## Overview

DynaMod-Diff generates industrial anomaly images from:

- a normal reference image;
- a binary defect mask;
- a text caption describing the object and defect type;
- a trained object-specific checkpoint.

The repository contains scripts for:

- MVTec AD dataset preparation and few-shot splitting;
- caption generation;
- model training;
- single-sample inference;

## Repository Structure

```text
DynaMod-Diff/
├── configs/
│   └── configs.yaml
├── dataset/
├── grounding_input/
├── ldm/
├── utils/
├── Infer_DATA/
│   └── mask/
├── process_Mvtec_per-object_json.py
├── convert_ckpt.py
├── main.py
├── trainer.py
├── infer.py
├── infer_batch.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhangzhijie2333/DynaMod-Diff.git
cd DynaMod-Diff
```

### 2. Create the environment

```bash
conda create -n DynaMod-Diff python=3.9 -y
conda activate DynaMod-Diff
pip install -r requirements.txt
```

Please install a PyTorch version compatible with the CUDA version on your machine.

### 3. Prepare the pretrained model

Download the Stable Diffusion v1.4 checkpoint and place it in a local directory, for example:

```text
sd-v1-4.ckpt
```

Download the complete `clip-vit-large-patch14` model folder and place it under:

```text
openai/
└── clip-vit-large-patch14/
```

Download sem_diffusion_pytorch_model.bin and place it under:

```text
model_sem/
└── sem_diffusion_pytorch_model.bin
```

Set the pretrained checkpoint path in the corresponding training script.

## Dataset Preparation

### 1. Download MVTec AD

Download the MVTec AD dataset and organize it using its original structure:

```text
mvtec_anomaly_detection/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
└── ...
```

### 2. Few-shot anomaly split

For each object category and defect type, one third of the available real anomaly image-mask pairs are selected as the few-shot training set. 

The resulting prepared data can follow this structure:

```text
DATA/Mvtec_per-object/
├── train/
│   ├── bottle/
│   │   ├── Ground_truth/
│   │   ├── Source_Images/
│   │   └── caption.json
│   ├── cable/
│   │   ├── Ground_truth/
│   │   ├── Source_Images/
│   │   └── caption.json
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   └── ...
└── test/
    ├── bottle/
    │   ├── Ground_truth/
    │   ├── Source_Images/
    │   └── caption.json
    ├── cable/
    │   ├── Ground_truth/
    │   ├── Source_Images/
    │   └── caption.json
    ├── capsule/
    ├── carpet/
    ├── grid/
    ├── hazelnut/
    └── ...

## Caption Generation

The script `process_Mvtec_per-object_json.py` is used only to generate
`caption.json` files for the already prepared training and testing subsets.

Run:

```bash
python process_Mvtec_per-object_json.py

## Configuration

The main configuration file is:

```text
configs/configs.yaml
```


## Training

Configure the dataset path, object category, defect types, pretrained checkpoint, and output directory.

Start training with:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

Training checkpoints and logs are saved to the output directory specified in the configuration.


## Inference

### Single-sample inference

Set the checkpoint path, reference image, mask, caption, output path, sampling steps, and guidance scale in `infer.py`.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py
```

## Citation

Please cite the corresponding paper when using this repository:

```bibtex
@article{zhang2026dynamoddiff,
  title   = {DynaMod-Diff: Dynamic Parameter Modulation for Industrial Anomaly Image Generation},
  author  = {Zhang, Zhijie and others},
  year    = {2026}
}
```

The citation information will be updated after publication.

## License

Please add an appropriate open-source license before public release. The licenses and usage restrictions of Stable Diffusion, MVTec AD, and all third-party dependencies must also be followed.
