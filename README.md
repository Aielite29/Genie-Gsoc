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

---

### Method 2: Advanced Physics-Informed Graph Contrastive Learning

In this approach, we enhance quark/gluon classification by leveraging graph-based representations enriched with domain-specific physics insights. The key components of this method are:

- **Graph Representations & Physics Features:**  
  Instead of relying solely on image data, we extract graph representations from jets. Each graph encodes the structure of energy deposits as nodes, with edges representing their physical connectivity. In addition to basic physics features such as transverse momentum (`pt`) and jet mass (`m0`), we introduce an **explosion metric**—computed as the ratio of the number of edges to the number of nodes. This metric captures the “explosiveness” of a jet, with gluon jets typically exhibiting denser connectivity due to their larger color charge.

- **Graph Neural Network Architecture:**  
  We employ a Graph Attention Network (GAT) based encoder with dense (skip) connections to capture both low-level and high-level features by fusing multi-scale information. This encoder is coupled with:
  - A **Projection Head** that projects the learned graph embeddings into a latent space suitable for contrastive learning.
  - A **Classifier Head** that concatenates the graph-level embedding with the global physics features—including `pt`, `m0`, and the explosion metric—to perform the final classification.

- **Contrastive Pre-training & Fine-tuning:**  
  The model undergoes a two-step training process:
  1. **Contrastive Pre-training:**  
     The encoder is pre-trained using an improved NT-Xent loss enhanced with margin-based hard negative regularization. This helps the model learn robust, discriminative representations by contrasting different augmented views of the same graph.
  2. **Classification Fine-tuning:**  
     After pre-training, the classifier head is fine-tuned (with an option to freeze the encoder) to optimize the final classification performance.

- **Complementarity with Image-Based Features:**  
  When used in tandem with traditional image-based approaches—such as those leveraging difference images between original and reconstructed data—the graph-based features provide complementary insights. They capture the connectivity dynamics and substructure of jets, which are critical for differentiating between the more focused quark jets and the denser, more “explosive” gluon jets.

- **Performance Metrics:**  
  When trained on only 10,000 images, this method achieved an Accuracy of **66.95%**, an F1 Score of **0.6678**, and a ROC-AUC of **0.7149**. These results demonstrate the potential of integrating graph-based, physics-informed features to enhance quark/gluon classification performance even with limited training data.

---

This advanced method leverages the underlying physics of jet formation and explosion dynamics to enhance anomaly detection and improve the robustness of quark/gluon classification.
![mermaid-diagram-2025-04-07-222319](https://github.com/user-attachments/assets/b7d155f6-c44b-4b57-b65e-8bad6d9c4e33)

--- 
---

## Results & Conclusion

- **Image Reconstruction:**  
  Implemented and optimized multi-channel autoencoders with attention mechanisms for enhanced image reconstruction.

- **Graph-Based Jet Representation:**  
  Developed a robust graph-based representation of jet images using dynamic kNN and physics-informed features. This included the novel **explosion metric** (ratio of the number of edges to nodes) which captures the connectivity dynamics of jets, providing deeper insights into quark and gluon substructures.

- **Scalability:**  
  Successfully handled large-scale datasets by splitting them into 14 chunks and processing them sequentially on Kaggle to avoid memory issues.

- **Three Classification Approaches:**

  1. **Baseline Approach:**  
     CNN-based models utilizing channel-wise convolutions, feature pyramids, and new tabular features derived from ECAL and HCAL channels. An ensemble of gradient-boosted tree (GBT) models—including LightGBM, XGBoost, and CatBoost—was employed to enhance the final predictions.

  2. **Enhanced Approach:**  
     Combined the difference images (original – reconstructed) with graph representations from Task 2 as input to convolutional neural networks. This method leverages complementary features from both the image and graph domains for improved feature extraction and classification.

  3. **Third Approach: Advanced Physics-Informed Graph Contrastive Learning:**  
     This method leverages physics-informed graph representations to enhance quark/gluon classification. Key features include:
     - **Contrastive Pre-training:** The encoder is pre-trained using an improved NT-Xent loss with margin-based hard negative regularization, which helps learn robust graph embeddings.
     - **Physics-Informed Features:** Graph representations are enriched with domain-specific features such as transverse momentum (`pt`), jet mass (`m0`), and the explosion metric (capturing the jet's connectivity dynamics).
     - **Fine-Tuning:** A classifier head is fine-tuned (with an option to freeze the encoder) on these combined features for final classification.
     
     When trained on only 10,000 images, this approach achieved an Accuracy of **66.95%**, an F1 Score of **0.6678**, and a ROC-AUC of **0.7149**. These metrics highlight the potential of integrating physics-informed graph features to enhance classification performance even with limited training data.

Overall, the combination of these approaches demonstrates a comprehensive strategy for quark/gluon classification. While the baseline and enhanced methods capitalize on advanced CNN architectures and image-based feature extraction, the third approach introduces a novel graph-based perspective grounded in the underlying physics of jet formation, leading to improved robustness and accuracy in the classification task.

---

---

## Contact

For any questions or clarifications, feel free to reach out:

- **GitHub:** [Aielite29](https://github.com/Aielite29)
- **LinkedIn:** [Abhinav Jha](https://www.linkedin.com/in/abhinav-jha-81ab8530b/)

---



