# GENIE Task - GSoC Submission

## Overview

This repository contains my submission for the **GENIE task** as part of **Google Summer of Code (GSoC)**. The project involves three major tasks:

1. **Image Reconstruction using Autoencoders** (Common Task 1)  
2. **Graph Construction and Processing** (Common Task 2)  
3. **Quark/Gluon Classification** (Specific Task)

The dataset consists of high-dimensional images representing quark/gluon explosions with three distinct channels:

- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

Due to the **large size** of the dataset, loading everything at once caused **Kaggle to go out of memory**. Therefore, I **divided the dataset into 14 chunks**, each containing approximately 140,000 images, and processed them **chunk by chunk** to avoid OOM issues.

---

## Task 1: Image Reconstruction using Autoencoders

The goal of this task was to **reconstruct images** using autoencoder models.

### Methodology

- Trained **three separate autoencoders**, one for each channel (**ECAL, HCAL, Tracks**).  
- Implemented a **self-attention block** to learn interdependencies between channels.  
- Developed **three pairwise attention mechanisms** (ECAL-HCAL, ECAL-Tracks, HCAL-Tracks) and a **global attention** mechanism to focus on all three channels simultaneously.  
- Reconstructed images using these **attention-based autoencoders**.  
- Processed the data **chunk by chunk** to manage memory usage on Kaggle.

### Reconstruction Metrics (Single Chunk)

The following metrics were computed on a single chunk of ~10,000 images:

#### Overall Metrics

| Metric                          | Value                    |
|---------------------------------|--------------------------|
| Average MSE                     | 6.57834425510373e-06      |
| Average MAE                     | 7.680848648305982e-05     |
| Average PSNR                    | 43.1297056164595         |
| Average SSIM                    | 0.987662136554718        |
| Average Energy Difference       | 1.3378345966339111       |

#### Per-Channel Metrics

| Channel | MSE           | MAE           | PSNR         | SSIM       |
|---------|---------------|---------------|--------------|------------|
| 1       | 1.5025887e-05 | 9.8354700e-05 | 39.14618492  | 0.97488469 |
| 2       | 4.4442954e-06 | 7.4819196e-05 | 38.30827117  | 0.97431507 |
| 3       | 2.6482948e-07 | 5.7251895e-05 | 24.96852008  | 0.63355858 |





![Editor _ Mermaid Chart-2025-04-01-180513](https://github.com/user-attachments/assets/154652ca-2dff-4b26-aa45-d48ced34f51b)





---

## Task 2: Graph Construction and Processing

In this task, jet images are transformed into graph representations using a multi-step pipeline:

- **Graph Conversion:**  
  Each jet image (from key **"X_jets"**) is converted into a graph where nodes represent active pixels (above a specified energy threshold). For each node, features such as total energy, individual channel energies (ECAL, HCAL, Tracks), pT fraction, charged fraction, local energy sum, and positional information (converted to (η, φ)) are computed.

- **Edge Formation:**  
  Two approaches are used to form edges between nodes:
  1. **Dynamic kNN:** A dynamic adjacency is computed using a weighted combination of geometric (η, φ) and feature-based distances, enabling adaptive connectivity.
  2. **Multi-Scale kNN:** Two neighborhood graphs are built at different scales (small and large k values) to capture both local and global relationships.

- **Coordinate Embedding:**  
  Learnable coordinate embeddings are applied to the (η, φ) values to enhance spatial feature learning.

- **Graph Neural Network (GNN):**  
  The processed graphs are then fed into a GNN that alternates between GAT and EdgeConv layers, optionally using SAGPooling for hierarchical feature extraction. The final model performs multi-scale pooling (mean, max, sum) and incorporates global features for robust downstream tasks.

- **Visualization:**  
  Sample graphs are visualized to illustrate the evolution from the normal kNN approach to dynamic and multi-scale kNN-based graphs processed by the GNN.

![Screenshot 2025-04-01 170707](https://github.com/user-attachments/assets/221ddc24-7f10-4e56-8bb7-f12139751d78)




![Editor _ Mermaid Chart-2025-04-01-181843](https://github.com/user-attachments/assets/3e02a425-d2be-4380-9769-72964384869f)
---

## Task 3: Quark/Gluon Classification

The final task was to classify images as **quarks** or **gluons**.

### Methodology

1. **Baseline Approach: CNN-Based Models**  
   - **Training multiple convolutional neural networks (CNNs)** including architectures such as **ResNet, DenseNet, EfficientNet, ViT, and Swin Transformer**.  
   - Leveraged **channel-wise convolutions** and **feature pyramids** for hierarchical feature extraction.  
   - Incorporated both **ECAL** and **HCAL** channels from the images to create new columns in the dataset that capture properties differing between quarks and gluons.  
   - Employed a **soft ensemble** of all CNN models for better generalization.

2. **Enhanced Approach:**  
   - To improve upon the baseline, the difference between the **original** and **reconstructed** images was computed.  
   - Additionally, graph representations of the images (generated in Task 2) were also utilized.  
   - These difference images and graph features were **concatenated** and used as a whole input to convolutional neural networks, allowing the model to learn complementary features for classification.

3. **Ensemble with Gradient Boosted Trees (GBTs):**  
   - Finally, multiple GBT models (including **LightGBM (LGBM), XGBoost (XGB), and CatBoost**) were trained on the extracted features.  
   - An **ensemble** of these GBT models was used to obtain the final classification predictions.
![Editor _ Mermaid Chart-2025-04-01-181147](https://github.com/user-attachments/assets/cc4d7e38-7696-4746-b67f-00fe0a6b2399)

---

## Results & Conclusion

- Implemented and optimized **multi-channel autoencoders** with attention mechanisms for enhanced image reconstruction.
- Developed a **robust graph-based representation** of jet images using dynamic kNN and physics-informed features.
- Achieved **high classification accuracy** by combining CNN-based feature extraction with gradient-boosted tree models.
- Successfully handled **large-scale datasets** by splitting them into **14 chunks** and processing them **chunk by chunk** to avoid memory issues on Kaggle.
- Two classification approaches were implemented:
  1. **Baseline:** CNN-based models using channel-wise convolutions, feature pyramids, and new tabular features derived from ECAL and HCAL channels, followed by an ensemble of GBTs.
  2. **Enhanced:** Combining the difference images (original – reconstructed) and graph representations from Task 2 as input to CNNs for improved feature extraction.

---

## Contact

For any questions or clarifications, feel free to reach out:

- **GitHub:** [Aielite29](https://github.com/Aielite29)
- **LinkedIn:** [Abhinav Jha](https://www.linkedin.com/in/abhinav-jha-81ab8530b/)

---



