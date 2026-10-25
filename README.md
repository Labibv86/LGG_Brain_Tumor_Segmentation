# LGG Brain Tumor Segmentation with U-Net

## Introduction

This project implements an end-to-end deep learning pipeline for brain tumor segmentation on the LGG MRI Segmentation Dataset. The goal is to predict pixel-level masks for low-grade glioma tumors from 2D FLAIR MRI slices, moving beyond standard image classification into dense semantic segmentation.

## Project Description

The pipeline is built in PyTorch and centered on a standard U-Net architecture with four encoder and four decoder blocks, connected by skip connections that preserve spatial detail for accurate tumor boundary detection. To address the severe class imbalance between tumor and background pixels, training uses a combined Dice and binary cross-entropy loss. Data is split at the patient level rather than the slice level, ensuring that no patient appears in both training and test sets, which prevents data leakage and gives an honest measure of generalization to unseen patients. The model is trained for 100 epochs with the Adam optimizer and a ReduceLROnPlateau scheduler that adjusts the learning rate based on validation Dice score.

On the held-out test set of 387 slices from 11 unseen patients, the model achieves a Dice coefficient of 0.8073, an IoU of 0.7764, a precision of 0.9293, and a recall of 0.8241. The high precision indicates the model rarely predicts false positives, while the lower recall suggests some tumor boundaries and smaller lesions are missed. Training curves, prediction overlays, and a full analysis are included in the project report.
