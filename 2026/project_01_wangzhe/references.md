# References

## Core Paper

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
> Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
> Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words:
> Transformers for Image Recognition at Scale.* ICLR 2021.
> https://arxiv.org/abs/2010.11929

The ViT implementation in `src/vit.py` is an educational from-scratch
re-implementation based on the architecture described in this paper.

## Third-Party Code

### torchvision Vision Transformer (ViT-B/16)

The pretrained comparison baseline referenced in the report uses
`torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`.
This is the official torchvision implementation distributed under the
BSD 3-Clause License:

- Repository: https://github.com/pytorch/vision
- Pretrained weights: provided by the PyTorch team

### Imagenette Dataset

- Curated by Jeremy Howard / fast.ai
- Downloaded from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
- Repository: https://github.com/fastai/imagenette

Imagenette is an ImageNet-derived subset. ImageNet images are subject to
their own terms of use (http://www.image-net.org/).

## Techniques Referenced

| Technique | Source |
|-----------|--------|
| Label Smoothing | Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", CVPR 2016 |
| Cosine Annealing LR | Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017 |
| RandAugment | Cubuk et al., "RandAugment: Practical automated data augmentation...", NeurIPS 2020 |
| Random Erasing | Zhong et al., "Random Erasing Data Augmentation", AAAI 2020 |
| MixUp | Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018 |
| CutMix | Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers...", ICCV 2019 |
| DropPath (Stochastic Depth) | Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016 |
| Linear Warmup | Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv 2017 |

## Tools & Libraries

- **PyTorch** — deep learning framework (BSD license)
- **torchvision** — datasets, transforms, pretrained models (BSD 3-Clause)
- **matplotlib** — figure generation (PSF-based license)
- **Pillow** — image loading (Historical Permission Notice and Disclaimer)
- **NumPy** — numerical computing (BSD license)
