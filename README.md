# Enhanced TransUNet : Integrating Dual-Attention-and-CutMix for Medical Image Segmentation

This repository is part of a final project for the "Deep Learning for Advanced Computer Vision 224C" course at the University of California, Santa Cruz. It reproduces and enhances the TransUNet model, as detailed in the paper ["TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"](https://arxiv.org/pdf/2102.04306). The project's initial phase involved training a baseline TransUNet model, followed by targeted modifications to improve its segmentation performance.

_Authors : Marian Zlateva (mzlateva@ucsc.edu) and Marzia Binta Nizam(manizam@ucsc.edu)_
<br><br>
_Details of our experiments can be found [here](https://github.com/marzianizam/Enhanced_TransUNet-Integrating-Dual-Attention-and-CutMix-for-Medical-Image-Segmentation/blob/main/Enhanced_TransUNet___Integrating_Dual_Attention_and_CutMix_for_Medical_Image_Segmentation.pdf)_

# Overview
This study focuses on enhancing the TransUNet model for medical image segmentation by improving its generalization capabilities. Initially, the baseline TransUNet model achieved a Mean Dice score of 0.769 and a median Hausdorff Distance of 32.87. By incorporating Channel Attention, Dual Attention mechanisms, and CutMix data augmentation, significant improvements were made, culminating in a Mean Dice score of **0.823** and a reduced median Hausdorff Distance of **19.74**. These strategic modifications have enhanced the model's ability to accurately segment complex anatomical structures, advancing the application of medical imaging.

## Results Summary

The following table summarizes the performance improvements across various organs:

| Model                          | Average Dice | Median HD95 | Aorta  | Gallbladder | Kidney (L) | Kidney (R) | Liver  | Pancreas | Spleen | Stomach |
|--------------------------------|--------------|-------------|--------|-------------|------------|------------|--------|----------|--------|---------|
| TransUNet  (Baseline)           | 0.769        | 32.87       | 0.868  | 0.596       | 0.814      | 0.740      | 0.945  | 0.542    | 0.873  | 0.778   |
| TransUNet  (ours)               | **0.823**        | **19.74**       | **0.882**  | **0.631**       | **0.860**      | **0.831**      | **0.946**  | **0.693**   | **0.907**  | **0.833**   |

*Note: All values are rounded to three decimal places for clarity.*


Below are the segmentation visualizations comparing the baseline and enhanced models:

![Segmentation Comparison](https://github.com/marzianizam/Enhanced_TransUNet-Integrating-Dual-Attention-and-CutMix-for-Medical-Image-Segmentation/blob/main/Result_Viz.png)

*Figure 1: Visual comparison of segmentation performance between baseline TransUNet and TransUNet with Dual Attention and CutMix.*


# Environment Setup

Please prepare an environment with python=3.7, and then use the command (following the original TransUnet repo)

```bash
pip install -r requirements.txt
```

You can view our environment specification [here](https://github.com/marzianizam/Enhanced_TransUNet-Integrating-Dual-Attention-and-CutMix-for-Medical-Image-Segmentation/blob/main/our_environment_spec.txt).

# Download Google pre-trained ViT models

Download the Google pre-trained ViT models following the original repository's instruction.

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ./model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ./model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

# Data

The experiments were conducted on the Synapse multi-organ segmentation dataset. Please refer to the [original repository](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) for the data preparation. 

 # Train

 Run the train script on synapse dataset. We used batch size 8 due to our limited GPU access, but the original code supports multiple GPUs as well.

 ```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

# Test

Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes. 

 ```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16 --is_savenii
```
You can download our trained model from [here](https://drive.google.com/file/d/1WJNikz0zN0X5WFQROPvMMC3wsk5wxXZy/view?usp=sharing) to test. 

# Visualization

Please refer to this [notebook](https://github.com/marzianizam/Enhanced_TransUNet-Integrating-Dual-Attention-and-CutMix-for-Medical-Image-Segmentation/blob/main/visualization.ipynb) for visualizing the predictions.

# Reference

* [TransUNet](https://github.com/Beckschen/TransUNet/tree/main)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet/tree/main)


# Acknowledgement

We appreciate the developers of [TransUNet](https://github.com/Beckschen/TransUNet/tree/main) and the provider of the [Synapse multi-organ segmentation](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) dataset. We are grateful to Professor [Yuyin Zhou](https://campusdirectory.ucsc.edu/cd_detail?uid=yzhou284) for her invaluable guidance and insightful suggestions throughout the duration of this project. :smiley: :smiley:


