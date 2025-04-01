# GENIE Task - GSoC Submission

## Overview

This repository contains my submission for the **GENIE task** as part of **Google Summer of Code (GSoC)**. The project involved three tasks:

1. **Image Reconstruction using Autoencoders** (Common Task 1)  
2. **Graph Construction and Processing** (Common Task 2)  
3. **Quark/Gluon Classification** (Specific Task)

The dataset consists of high-dimensional images representing quark/gluon explosions, with three distinct channels:

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

### Sample Reconstruction Statistics (Single Chunk)

Below are **reconstruction metrics** computed on a **single chunk** of ~10,000 images:

```
Average MSE:                6.57834425510373e-06
Average MAE:                7.680848648305982e-05
Average PSNR:               43.1297056164595
Average SSIM:               0.987662136554718
Average Energy Difference:  1.3378345966339111
Average Correlation Coefficient: -0.0007801791299508257

Per-channel MSE:            [1.5025887e-05 4.4442954e-06 2.6482948e-07]
Per-channel MAE:            [9.8354700e-05 7.4819196e-05 5.7251895e-05]
Per-channel PSNR:           [39.14618492 38.30827117 24.96852008]
Per-channel SSIM:           [0.97488469 0.97431507 0.63355858]
```

---

## Task 2: Graph Construction and Processing

This task involved **converting jet images into graphs** and extracting meaningful physics-informed features.

### Key Components

1. **Dynamic kNN Class**  
   - Recomputes adjacency based on a weighted sum of geometric distance (η, φ) and feature distance in the node embedding space.

2. **Multi-Scale kNN**  
   - Constructs multi-scale graphs for hierarchical feature learning.

3. **Jet Graph Processor**  
   - Converts a jet image into a graph while incorporating important node features such as:
     - Total energy
     - ECAL, HCAL, Tracks
     - **pT fraction, charged fraction**
     - Local energy density (3×3 sum)
     - Coordinates (η, φ)
     - **Log(E+1), sqrt(E), angle, normalized distance**, etc.

4. **Coordinate Embedding Class**  
   - Implements learnable embeddings for (η, φ) coordinates to enhance spatial feature learning.

5. **Graph Neural Network (GNN)**  
   - **Coordinate embedding** for (η, φ) at specific indices.
   - **Dynamic adjacency module** for continuous graph refinement.
   - **GNN layers** alternating between **GAT (Graph Attention Network)** and **EdgeConv**.
   - **Optional SAGPooling** for hierarchical pooling.
   - Final **multi-scale pooling classifier** (mean, max, sum) combined with global features.

### Graph Evolution

Below are **visualizations** of how the graph evolves through different processing stages (example images shown):

#### **Normal Knn approach**
![Screenshot 2025-04-01 170559](https://github.com/user-attachments/assets/04170f35-de57-48ca-b242-1e032019a32d)


#### **Dynamic Knn + GNN**
![Screenshot 2025-04-01 170651](https://github.com/user-attachments/assets/443f21bf-1d0b-48a3-8edf-8b98ba6397f3)


#### **Dynamic KNN + MultiScale KNN +GNN with GAT**
![Screenshot 2025-04-01 170707](https://github.com/user-attachments/assets/7992e586-a5d9-4ae3-8831-4bc341a8605f)


---

## Task 3: Quark/Gluon Classification

The final task was to classify images as **quarks** or **gluons**.

### Methodology

1. **CNN-Based Models**  
   - Trained multiple **convolutional neural networks (CNNs)**, including **ResNet, DenseNet, EfficientNet, ViT, Swin Transformer**.  
   - Incorporated **channel-wise convolutions** and **feature pyramids** for hierarchical feature extraction.  
   - Used a **soft ensemble** of all CNN models for better generalization.

2. **Tabular Data Augmentation**  
   - Created a new feature column in tabular data using the **CNN predictions**.  
   - Computed additional **quark/gluon properties** from the provided dataset.

3. **Gradient Boosted Trees (GBTs)**  
   - Trained multiple **GBTs** on the tabular dataset: **LightGBM (LGBM), XGBoost (XGB), CatBoost**.  
   - Took an **ensemble** of all GBT models to obtain the final classification predictions.

4. **Alternative Approach: Reconstructed Image Differences + Graph-Based Classification**  
   - Analyzed **difference images** (original – reconstructed) to highlight distribution differences for quarks vs. gluons.  
   - Combined **difference images** and **graph representations** from Task 2.  
   - Incorporated **channel-wise convolutions** and **feature pyramids**.  
   - Due to **resource constraints**, was unable to fully train this model for many epochs.

---

## Results & Conclusion

- Implemented and optimized **multi-channel autoencoders** with attention mechanisms for **better image reconstruction**.
- Developed a **robust graph-based representation** of jet images using dynamic kNN and physics-informed features.
- Achieved **high classification accuracy** by combining CNN-based feature extraction with gradient-boosted tree models.
- Successfully handled **large-scale datasets** by splitting into **14 chunks** and **processing chunk by chunk** to avoid memory issues on Kaggle.
- Proposed **two classification approaches**:
  1. **CNN-based models** with tabular feature augmentation and ensemble learning.
  2. **Reconstructed image differences** and **graph-based classification** using physics-informed graphs.
- **Full execution** of the second approach was limited by computational constraints but the **code is implemented**.

---

## Contact

For any questions or clarifications, feel free to reach out:

- **GitHub:** [Aielite29](https://github.com/Aielite29)
- **LinkedIn:** [Abhinav Jha](https://www.linkedin.com/in/abhinav-jha-81ab8530b/)

---




