# Deep Ensemble with Channel-wise FPN for Image Classification

## System Architecture

The architecture is divided into two primary components:

1. **Deep Learning Ensemble:**
   - **Model Ensemble:** Five convolutional network architectures form the backbone of the deep ensemble.
   - **Channel-wise FPN Module:** Each network is augmented with a channel-wise FPN that applies multiple convolution operations with different kernel sizes, enhancing feature richness.
   - **Feature Fusion:** The modified deep networks produce enriched feature maps that serve as the basis for classification.

2. **Metadata-based Gradient Boosting:**
   - **Feature Engineering:** Physics-based metadata features are computed directly from the image channels, such as ratios, mean values, and differences.
   - **Gradient Boosting Models:** Multiple gradient boosting models (e.g., XGBoost, CatBoost, and a Histogram-based Gradient Boosting classifier) are trained on the engineered metadata.
   - **Ensemble Prediction:** The predictions from the deep learning ensemble and the gradient boosting ensemble are averaged to produce the final classification output.

---

## Methodology

### Deep Convolutional Networks

The framework leverages five pre-trained models:
- **ResNet18:** Renowned for its residual learning capabilities.
- **DenseNet121:** Uses dense connections to promote feature reuse.
- **EfficientNetB0:** Strikes a balance between accuracy and computational efficiency.
- **Vision Transformer (ViT):** Applies transformer architecture principles to image patches.
- **Swin Transformer:** Implements hierarchical vision transformer principles.

Each model is adapted to accept enhanced feature maps from the channel-wise FPN module, ensuring rich representation of the input images.

### Channel-wise Feature Pyramid Network (FPN)

The custom channel-wise FPN processes each image channel independently by applying three different convolution operations (1x1, 3x3, and 5x5). The resulting features are fused to yield an enriched representation, capturing both fine-grained and coarse spatial information. This enhancement facilitates more robust feature extraction across the deep models.

### Metadata Feature Engineering

Beyond deep visual features, the system computes physics-based metadata directly from the input images. These engineered features include:
- **Channel Ratios:** Mean ratios calculated between the channels.
- **Mean Values:** Average intensity values per channel.
- **Differences:** Mean differences and normalized absolute differences between channels.

These additional features provide critical supplementary context that enhances the overall predictive power of the ensemble.

### Gradient Boosting Ensemble

To further refine predictions:
- **Training:** Three gradient boosting models are trained using the engineered metadata.
- **Ensemble Strategy:** The predictions from these models are averaged.
- **Final Integration:** The averaged metadata predictions are combined with the deep ensemble predictions to form the final output, ensuring improved robustness and accuracy.

---


---


## Evaluation

The framework's performance is evaluated using standard classification metrics such as accuracy and log loss. Detailed evaluation scripts are included to facilitate the analysis on validation and test datasets. The multi-stage ensemble approach emphasizes both robustness and reliability, making it suitable for applications where both image content and auxiliary metadata are critical.

---
