# Deep Ensemble with Channel-wise FPN for Image Classification

This repository implements a deep ensemble framework that significantly enhances image classification performance by integrating graph-based features and the difference between original and reconstructed images directly within the deep learning pipeline. These additions are embedded into the deep models to improve feature extraction and highlight subtle yet critical discrepancies in the input data.

---

## System Architecture

### Deep Learning Ensemble

The core of the framework is an ensemble of five state-of-the-art deep learning models:

- **ResNet18**
- **DenseNet121**
- **EfficientNetB0**
- **Vision Transformer (ViT)**
- **Swin Transformer**

Each model is augmented with a custom channel-wise Feature Pyramid Network (FPN) module, which applies multiple convolution operations (1x1, 3x3, and 5x5) on each image channel independently. This setup generates enriched feature maps that are fed into the base networks.

### Enhanced Feature Extraction: Graph and Reconstructed Image Differences

Two key enhancements are incorporated directly into the deep learning part of the framework:

1. **Graph Features:**
   - **Purpose:** Graph features are designed to capture the spatial structure and connectivity within an image. By modeling the image as a graph, where nodes represent regions or pixels and edges capture relationships, the network can leverage global and local structural information.
   - **Impact:** This additional structural understanding allows the deep models to detect complex patterns and relationships that standard convolutional operations might overlook. It contributes to a more nuanced feature representation, especially in images with intricate spatial patterns.

2. **Reconstructed Image Differences:**
   - **Purpose:** The framework computes the difference between the original image and its reconstructed version. This process is executed exactly as shown in the code and is integrated into the deep learning pipeline.
   - **Impact:** The difference image effectively emphasizes subtle discrepancies, noise, and anomalies. These deviations often indicate important variations in the image that are critical for accurate classification. By explicitly modeling these differences, the network becomes more sensitive to fine-grained details, leading to improved discriminative performance.

---

## Methodology

### Deep Convolutional Networks

The framework leverages five pre-trained models, each adapted to handle enhanced feature maps produced by the channel-wise FPN. The models benefit from their unique architectural strengths:

- **ResNet18** leverages residual connections to facilitate deeper network training.
- **DenseNet121** uses dense connectivity to encourage feature reuse.
- **EfficientNetB0** optimizes the balance between accuracy and efficiency.
- **Vision Transformer (ViT)** and **Swin Transformer** apply transformer principles to capture long-range dependencies in image data.

### Channel-wise Feature Pyramid Network (FPN)

The channel-wise FPN module independently processes each image channel using three convolution operations (1x1, 3x3, and 5x5). These multi-scale operations capture diverse spatial features:
- **Fine-grained details** are captured by smaller kernels.
- **Broader contextual information** is acquired through larger kernels.
- **Fusion:** The outputs are fused to form a rich representation that is passed on to the deep learning models.

### Integration of Graph Features and Reconstructed Image Differences

The incorporation of graph features and the reconstructed image difference plays a pivotal role in enhancing the deep models:

- **Graph Features:**
  - By constructing a graph representation of the image, the model accesses information about the spatial relationships and connectivity between different image regions. This insight is especially valuable when distinguishing between classes that differ in structural characteristics.
  - **Example Impact:** In complex scenes or medical images, where subtle structural variations are crucial, graph features help the network to focus on relational patterns, thereby increasing classification accuracy.

- **Reconstructed Image Differences:**
  - The process of subtracting a reconstructed image from the original highlights minute differences that are not easily detected by standard convolutions.
  - **Example Impact:** These differences can expose hidden noise or small variations that serve as vital clues for classifying images with minor anomalies or defects. This mechanism makes the network more robust to variations in illumination, texture, and other factors that typically degrade performance.

By embedding these enhancements into the deep learning pipeline, the framework achieves a more robust and discriminative feature representation. This directly translates into better performance, as the network can more effectively leverage both the explicit structural information and the subtle discrepancies within the input data.

---

