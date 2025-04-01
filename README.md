# GENIE Task - GSoC Submission

## Overview
This repository contains my submission for the GENIE task as part of Google Summer of Code (GSoC). The project involved three tasks:
1. **Image Reconstruction using Autoencoders** (Common Task 1)
2. **Graph Construction and Processing** (Common Task 2)
3. **Quark/Gluon Classification** (Specific Task)

The dataset consists of high-dimensional images representing quark/gluon explosions, with three distinct channels:
- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

Due to the large size of the dataset, I divided it into 14 chunks, each containing approximately 140,000 images.

---

## Task 1: Image Reconstruction using Autoencoders
The goal of this task was to reconstruct images using autoencoder models.

### Methodology:
- Trained **three separate autoencoders**, one for each channel (ECAL, HCAL, Tracks).
- Implemented a **self-attention block** to learn interdependencies between channels.
- Developed **three pairwise attention mechanisms** (ECAL-HCAL, ECAL-Tracks, HCAL-Tracks) and a **global attention** mechanism to focus on all three channels simultaneously.
- Reconstructed images using these attention-based autoencoders.

---

## Task 2: Graph Construction and Processing
This task involved converting jet images into graphs and extracting meaningful physics-informed features.

### Key Components:
#### 1. **Dynamic kNN Class**
   - Recomputes adjacency based on a weighted sum of geometric distance (η, φ) and feature distance in the node embedding space.

#### 2. **Multi-Scale kNN**
   - Constructs multi-scale graphs for hierarchical feature learning.

#### 3. **Jet Graph Processor**
   - Converts a jet image into a graph while incorporating important node features such as:
     - Total energy
     - ECAL, HCAL, Tracks
     - **pT fraction, charged fraction**
     - Local energy density (3x3 sum)
     - Coordinates (η, φ)
     - **Log(E+1), sqrt(E), angle, and normalized distance**

#### 4. **Coordinate Embedding Class**
   - Implements learnable embeddings for (η, φ) coordinates to enhance spatial feature learning.

#### 5. **Graph Neural Network (GNN)**
   - **Coordinate embedding for (η, φ)** at specific indices.
   - **Dynamic adjacency module** for continuous graph refinement.
   - **GNN layers** alternating between **GAT (Graph Attention Network)** and **EdgeConv**.
   - **Optional SAGPooling** for hierarchical pooling.
   - Final **multi-scale pooling classifier** (mean, max, sum) combined with global features.

---

## Task 3: Quark/Gluon Classification
The final task was to classify images as **quarks** or **gluons**.

### Methodology:
#### **1. First Approach: CNN-Based Models**
- Trained multiple **convolutional neural networks (CNNs)**, including:
  - **ResNet, DenseNet, EfficientNet, ViT, Swin Transformer**
- Incorporated **channel-wise convolutions** and **feature pyramids** for hierarchical feature extraction.
- Used a **soft ensemble** of all CNN models for better generalization.

#### **2. Tabular Data Augmentation**
- Created a new feature column in tabular data using the **CNN predictions**.
- Computed additional **quark/gluon properties** from the provided dataset.

#### **3. Gradient Boosted Trees (GBTs)**
- Trained multiple **GBTs** on the tabular dataset, including:
  - **LightGBM (LGBM)**
  - **XGBoost (XGB)**
  - **CatBoost**
- Took an ensemble of all GBT models to obtain the final classification predictions.

#### **4. Second Approach: Reconstructed Image Differences and Graph-Based Classification**
- Instead of directly classifying the images, another approach involved **subtracting the reconstructed images from the original images** to analyze the difference in distributions between quarks and gluons. The resulting distance differences were used as features for classification.
- Additionally, the **original images along with the difference images and the graph representations** created in Task 2 were used as inputs for classification.
- Both approaches incorporated **channel-wise convolutions** and **feature pyramids** to enhance feature extraction.
- However, due to **resource constraints**, I was unable to train the model for many epochs and could only implement the code without fully executing it.

---

## Results & Conclusion
- Implemented and optimized **multi-channel autoencoders** with attention mechanisms for better image reconstruction.
- Developed a **robust graph-based representation** of jet images using dynamic kNN and physics-informed features.
- Achieved **high classification accuracy** by combining CNN-based feature extraction with gradient-boosted tree models.
- Successfully handled **large-scale datasets** by chunking and parallelizing computations.
- Proposed **two classification approaches**: 
  1. **CNN-based models with tabular feature augmentation and ensemble learning.**
  2. **Reconstructed image differences and graph-based classification using physics-informed graphs.**
- Full execution of the second approach was limited by computational constraints.

This submission demonstrates a **multi-faceted approach to analyzing high-energy physics data**, leveraging deep learning, graph processing, and ensemble modeling techniques.

---

## Contact
For any questions or clarifications, feel free to reach out:
- **GitHub:** [Aielite29](https://github.com/Aielite29)
- **LinkedIn:** [Abhinav Jha](https://www.linkedin.com/in/abhinav-jha-81ab8530b/)

---





