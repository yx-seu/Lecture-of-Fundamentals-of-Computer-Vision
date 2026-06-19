# References and Attribution

## Libraries and Models

1. Hugging Face Transformers
   - Website: https://huggingface.co/docs/transformers/index
   - Usage in this project:
     - ADE20K checkpoint format compatibility
     - fallback loading for older exported model directories

2. SegFormer
   - Paper: Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers."
   - arXiv: https://arxiv.org/abs/2105.15203

3. ImageNet
   - Dataset page: https://www.image-net.org/
   - Usage in this project:
     - Target natural-image domain required by the course project.

4. ImageNet-S
   - Project page: https://github.com/LUSSeg/ImageNet-S
   - Usage in this project:
     - Recommended ImageNet-based dataset option when supervised segmentation masks are required.

5. PyTorch
   - Website: https://pytorch.org/

## Attribution Statement

This repository does not copy third-party training code verbatim. It uses public APIs from PyTorch, TorchVision, and Hugging Face Transformers to build a custom training, evaluation, inference, and visualization pipeline for the course project.

The final assignment model structure is implemented in `src/own_segformer.py`, including the encoder, decoder, and rebuilt classifier head.
