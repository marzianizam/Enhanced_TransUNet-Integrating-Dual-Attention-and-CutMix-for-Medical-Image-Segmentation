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

# Visualization

Please refer to thie notebook for visualizing the predictions.

# Reference

* [TransUNet](https://github.com/Beckschen/TransUNet/tree/main)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [DA-TransUnet](https://github.com/SUN-1024/DA-TransUnet/tree/main)


# Acknowledgement

We appreciate the developers of [TransUNet](https://github.com/Beckschen/TransUNet/tree/main) and the provider of the [Synapse multi-organ segmentation](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) dataset. We are grateful to Professor [Yuyin Zhou](https://campusdirectory.ucsc.edu/cd_detail?uid=yzhou284) for her invaluable guidance and insightful suggestions throughout the duration of this project. :smiley: :smiley:


