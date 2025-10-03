# ELE670 – Kvasir-Capsule

# Kvasir-Capsule Classification (Work in Progress)

This repository contains experiments on the **Kvasir-Capsule** dataset. The work so far includes exploratory data analysis (EDA) and an initial ResNet50 training pipeline.

## Notebooks

- **00_data_checks.ipynb** – basic dataset and metadata checks  
- **01_eda.ipynb** – exploratory data analysis (pixel intensity, RGB, PCA, texture features)  
- **03_resnet50.py** – initial ResNet50 training script (sanity check)  
- **03_resnet50_efficient.ipynb** – efficient ResNet50 training pipeline  
- **gpu_test.py** – test GPU availability  
- **pick_gpu.py** – select least-used GPU  
- **tiny_resnet50_best.pt** – example checkpoint from a small run  

## Progress

- Verified dataset structure and metadata  
- Performed EDA showing class imbalance and patient-specific clustering  
- Excluded very small classes (e.g. *Ampulla of Vater*, *Polyp*, *Blood–hematin*)  
- Implemented and tested efficient ResNet50 training pipeline  
  - 2-epoch sanity run: ~403s, GPU usage up to 80%  
- Next: extend to full ResNet50 model with cross-validation and improved performance
