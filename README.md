# 🧠 Brain Tumor Detection with YOLOv2 and Medical SAM

This project implements an automated brain tumor detection and segmentation system using a combination of YOLOv2 for object detection and Medical Segment Anything Model (Medical SAM) for segmentation.

## 📌 Overview

- **Object Detection**: YOLOv2 trained on annotated MRI tumor datasets
- **Segmentation**: Meta AI's Medical SAM used to precisely segment detected regions
- **Interface**: MATLAB App Designer GUI to select an image, detect tumor, and visualize segmentation

## 🔧 Technologies

- MATLAB R2024b
- Deep Learning Toolbox
- Computer Vision Toolbox
- Medical Segment Anything Toolbox

## 🚀 Usage
Clone the repo or download .mlapp and .mat files.

Open BrainTumorDetection.mlapp in MATLAB R2024b.

Click Load Custom Image and select a brain MRI image.

Click Detect to run YOLOv2 + MedicalSAM and view predictions and segmentation.

Note: Detection requires tumorDetector_fixed.mat to be present in the same folder.

## 👥 Contributors
[mustafayngl](https://github.com/mustafayngl)
[Xendoksia](https://github.com/Xendoksia)
[riadmmdli](https://github.com/riadmmdli)
[canomercik](https://github.com/canomercik)
[UmutSemihSoyer](https://github.com/UmutSemihSoyer)






