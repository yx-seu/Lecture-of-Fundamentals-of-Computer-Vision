#!/usr/bin/env bash
set -e
python -m src.infer \
  --model_dir outputs/imagenet_s_animals_10cls_trainval_holdout_ade/best_model \
  --input data/test_examples \
  --output results/demo_inference_10cls
