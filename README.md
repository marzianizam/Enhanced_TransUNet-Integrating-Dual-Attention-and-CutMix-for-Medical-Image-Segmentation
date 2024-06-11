# Enhanced TransUNet : Integrating Dual-Attention-and-CutMix for Medical Image Segmentation

This repository is part of a final project for the "Deep Learning for Advanced Computer Vision 224C" course at the University of California, Santa Cruz. It reproduces and enhances the TransUNet model, as detailed in the paper ["TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"](https://arxiv.org/pdf/2102.04306). The project's initial phase involved training a baseline TransUNet model, followed by targeted modifications to improve its segmentation performance.

_Authors : Marian Zlateva and Marzia Binta Nizam_

# Environment

Please prepare an environment with python=3.7, and then use the command (following the original TransUnet repo)

```bash
pip install -r requirements.txt
```

You can view our environment specification here.

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
You can download our trained model from here to test. 




