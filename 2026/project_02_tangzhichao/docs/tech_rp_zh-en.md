# Technical Summary: DINOv2-based Semantic Segmentation Model with Multi-Head ADBA Architecture
## Project Overview
This project addresses the limitations of traditional Vision Transformers (ViTs) in dense prediction tasks by proposing a semantic segmentation architecture built on a DINOv2 backbone and a custom Multi-Head ADBA-Head. By combining a multi-stage loss strategy with a semi-automated data augmentation pipeline, this work significantly improves segmentation accuracy and edge detail, particularly in scenarios with sparse annotations.

## 1. Backbone Selection: Unsupervised Spatial Priors via DINOv2
Limitations of Traditional ViTs: Conventional ViT models driven by classification labels tend to focus their attention excessively on the core regions of highly salient objects, often ignoring broader contextual and spatial relationships.

Introducing Unsupervised Priors: We selected DINOv2 (ViT-B/14 with registers) as our backbone. Thanks to its unsupervised pre-training paradigm, DINOv2 naturally learns to capture fine-grained geometric and semantic relationships between pixels without relying on manual labels. This robust prior knowledge provides an exceptionally stable and high-quality feature foundation for dense segmentation tasks.

## 2. Decoder Innovation: Multi-Head ADBA-Head
The decoder is designed with a multi-head attention diffusion and multi-scale feature fusion module to precisely recombine high-level abstract semantics with low-level spatial details.

Multi-Head Retention Mechanism: We directly utilize the attention map extracted from Layer 11 of the backbone to modulate the coarse, high-level features. Compared to our initial design, the current architecture strictly maintains the independent evolution and diffusion space of the multi-head features. This prevents cross-head averaging from destroying the rich representational knowledge already learned by the backbone.

Deep-Shallow Feature Fusion: Because segmentation is highly sensitive to edges and details, the model pulls shallow geometric features from Layer 3 via a non-intrusive hook. These are concatenated with the modulated high-level features. The fused representation undergoes deep convolution and is finally upsampled by exactly 14× using PixelShuffle combined with bilinear interpolation to generate the high-resolution segmentation bitmap.

## 3. Optimization Strategy: Multi-Stage Annealing Loss
The original training scheme suffered from two main pain points: a "lazy model" tendency (predicting everything as background) and blurry boundaries. To counter this, we introduced a composite loss design paired with a tailored training strategy:

Background and Boundary Penalties: To address the lazy tendency, we introduced an explicit classification loss for the background class. To solve edge blurring, we implemented a Boundary Loss computed from image gradients, forcing the model to yield sharper pixel-level predictions.

Multi-Stage Training: Introducing all losses at once initially caused gradient conflicts and non-convergence. Therefore, we split the training into two stages. Stage 1 disables the background and boundary penalties, allowing the model to quickly learn the main subjects and acquire basic segmentation capabilities. Stage 2 gradually enables these auxiliary losses to fine-tune the model, forcing it to refine edges and accurately identify background regions.

## 4. Data Pipeline: SAM-H Empowered Semi-Automated Annotation
The scarcity of dense annotations in the ImageNet-S dataset directly limited the learning ceiling of our custom decoder. To break through this data bottleneck, we built a semi-automated pseudo-label generation engine integrating the Segment Anything Model (SAM).

Closed-Loop Generation Process: 
1. Our partially-trained model performs forward inference on unannotated or weakly annotated images to output a coarse segmentation mask.
2. This coarse mask is converted into a Bounding Box (BBox), which serves as a prompt for SAM-H (which possesses stronger feature extraction capabilities).
3. SAM-H then generates a high-quality segmentation mask with precise edges.

Manual QC and Knowledge Distillation: The outputs from SAM-H undergo manual screening to filter out low-quality samples. The resulting high-precision pseudo-labels are fed back into the training set, creating an upward spiral of continuous improvement for both the dataset and the model's capabilities.