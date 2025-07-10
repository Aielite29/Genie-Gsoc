# GENIE Task - GSoC Submission

## Overview

This repository contains my submission for the **GENIE task** as part of **Google Summer of Code (GSoC)**. The project involves three major tasks:


1. **Image Reconstruction using Autoencoders** (Common Task 1)  
2. **Graph Construction and Processing** (Common Task 2)  
3. **Quark/Gluon Classification** (Specific Task 1)  
4. **Non-local GNN for Jet Classification** (Specific Task 4)
   
The dataset consists of high-dimensional images representing quark/gluon explosions with three distinct channels:

- **ECAL (Electromagnetic Calorimeter)**
- **HCAL (Hadronic Calorimeter)**
- **Tracks**

Due to the **large size** of the dataset, loading everything at once caused **Kaggle to go out of memory**. Therefore, I **divided the dataset into 14 chunks**, each containing approximately 10,000 images, and processed them **chunk by chunk** to avoid OOM issues.

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


-**Performance Note:**  
The model achieved an accuracy of **69.0124%** after training on **40,000 jet graphs over 6 epochs**. The original architecture was designed with the assumption of full dataset availability; as a result, training on a reduced subset led to suboptimal parameter convergence due to the model's complexity. To mitigate this, **dense connections were introduced within the GNN**, improving gradient flow and feature propagation, which led to the observed performance under limited data conditions.


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

## Method 2: Advanced Physics-Informed Graph Contrastive Learning

In this approach, we significantly enhance quark/gluon jet classification by leveraging graph-based representations enriched with domain-specific physics insights. This method is built upon an improved graph neural network architecture and a two-step training procedure (contrastive pre-training followed by classification fine-tuning). The complementary nature of these graph features further provides insights that augment traditional image-based approaches.

### 1. Graph Representations & Physics Features

- **Graph Extraction:**  
  Rather than relying solely on image data, we extract graph representations from jet event data. In these graphs, nodes encode energy deposits, and edges represent the physical connectivity among them.

- **Enhanced Physics Features:**  
  In addition to standard features such as transverse momentum (`pt`) and jet mass (`m0`), we introduce an **explosion metric**. This metric is computed as the ratio of the number of edges to the number of nodes and quantifies the “explosiveness” of a jet. Gluon jets, which typically exhibit denser connectivity owing to their larger color charge, tend to have higher explosion metric values.

- **Feature Normalization & Robustness:**  
  All global physics features are normalized to improve convergence and ensure robust learning. Safeguards are built into the dataset loader to handle missing or uninitialized node features.

---

### 2. Graph Neural Network Architecture

Our model is built around an enhanced **Graph Attention Network (GAT)** encoder with a modular design and improved robustness through contrastive and auxiliary learning. Key components include:

- **Enhanced GAT Encoder:**  
  - **Layer Normalization & Dropout:** Each GAT layer output is passed through `LayerNorm` and dropout to promote stable training and reduce overfitting.
  - **Residual Dense Connections:** Skip connections with linear projections of the input are added to the GAT outputs, helping preserve low-level features and enabling better feature fusion across layers.
  
- **Projection Head:**  
  A two-layer MLP that maps the final graph-level embedding into a latent space optimized by a contrastive loss. This latent space is designed to be invariant to graph augmentations, supporting robust self-supervised learning.

- **Auxiliary Reconstruction Head (Optional):**  
  A reconstruction module is attached to the encoder that predicts the **globally pooled node features** (using `global_mean_pool`). This encourages the encoder to retain richer node-level and structural information during pre-training. The reconstruction loss is computed using MSE.

- **Classifier Head:**  
  For supervised fine-tuning, a lightweight MLP classifier takes the concatenation of:
  - The graph-level embedding from the encoder,
  - Physics-motivated global features (`pt`, `m0`, and an explosion metric).  
  This setup allows the model to leverage both learned representations and domain-specific features for quark vs. gluon classification.

---

### 3. Contrastive Pre-training & Fine-tuning

The model undergoes a two-stage training process:

- **Stage 1: Contrastive Pre-training**
  - **Graph Augmentations:** Two stochastic graph views are created per sample using node dropout and edge perturbation.
  - **Improved NT-Xent Loss:**  
    We use a contrastive loss based on NT-Xent, augmented with margin-based **hard negative regularization**. This ensures that embeddings of different graphs are well-separated, while those of augmented views of the same graph remain close.
  - **Auxiliary Reconstruction Loss (Optional):**  
    When enabled, a reconstruction MSE loss is added between the predicted and actual **pooled node features**. This auxiliary objective promotes richer structural representations in the encoder.

- **Stage 2: Classification Fine-tuning**
  - The encoder (frozen or trainable) is paired with the classifier head.
  - A standard **cross-entropy loss** is used to fine-tune the model for binary classification of quark vs. gluon jets.

---

### 4. Complementarity with Image-Based Features

Our graph-based approach complements traditional image-based jet tagging methods. While image models excel at capturing spatial and calorimetric signatures, our graph representation emphasizes **connectivity patterns, relational structures, and substructure dynamics**—crucial for capturing the differences between collimated quark jets and more diffuse gluon jets. Fusion strategies combining both modalities can offer superior performance.



- **Performance Metrics:**  
  When trained on only 10,000 images, this method achieved an Accuracy of **66.95%**, an F1 Score of **0.6678**, and a ROC-AUC of **0.7212**. These results demonstrate the potential of integrating graph-based, physics-informed features to enhance quark/gluon classification performance even with limited training data.
![Editor _ Mermaid Chart-2025-04-07-234004](https://github.com/user-attachments/assets/a64f80ba-81a6-4265-9fa9-4582288a663e)

---

This advanced method leverages the underlying physics of jet formation and explosion dynamics to enhance anomaly detection and improve the robustness of quark/gluon classification.


--- 

# Task 4: Non-local GNN for Jet Classification

## Overview

In this task, we extend our jet classification pipeline by integrating a non-local graph neural network (GNN) module. The aim is to capture long-range dependencies among nodes in the graph representations of jet images, thereby enhancing the classification performance when distinguishing between quark and gluon jets.

## Motivation

Traditional GNNs operate on locally connected node neighborhoods through message passing (using dynamic kNN, GAT, EdgeConv, etc.). Although this approach successfully extracts local features, it may overlook global contextual information that can be critical for complex tasks like jet classification. The non-local block introduces a self-attention mechanism that allows every node to interact with every other node, capturing long-range dependencies and improving feature aggregation.

## Methodology

1. **Non-local Block Module:**  
   - We developed a non-local block that computes pairwise interactions between nodes using learned projections (θ, ϕ, and g).  
   - The block generates an attention map that captures global dependencies and applies it to aggregate features, incorporating a residual connection with the original node features.

2. **Integration into the GNN Pipeline:**  
   - The non-local block is inserted into the existing graph neural network architecture (which already employs dynamic graph construction, GAT, and EdgeConv layers) right after the local message-passing layers and before the pooling stage.  
   - This configuration is controlled by a flag (e.g., `use_non_local`) to seamlessly switch between the baseline local GNN and the enhanced non-local variant.

3. **Performance Comparison:**  
   - Both the baseline and non-local GNN models were trained under similar experimental conditions.  
   - Model performance was primarily evaluated using the ROC-AUC metric (in addition to accuracy, F1, etc.).  
   - The integration of the non-local block yielded a higher ROC-AUC score, demonstrating that capturing global dependencies plays a vital role in improving quark versus gluon classification.

## Results

For example:
- **Baseline GNN ROC-AUC:** **69.0124**
- **Non-local GNN ROC-AUC:** **71.0424**

These experimental findings support the hypothesis that non-local operations, through their ability to capture long-range interactions, contribute significantly to better performance in complex classification tasks such as jet classification.

![Editor _ Mermaid Chart-2025-04-08-004107](https://github.com/user-attachments/assets/9079e938-d5fc-4af1-a30d-4edfbef50fa5)

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
- **LinkedIn:** [Abhinav Jha](https://www.linkedin.com/in/abhinav-jha-0497a41ba/)

---



