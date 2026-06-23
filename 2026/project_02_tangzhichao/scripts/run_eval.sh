#!/usr/bin/env bash
set -e
python -m src.evaluate \
  --model_dir outputs/imagenet_s_animals_10cls_trainval_holdout_ade/best_model \
  --config configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml \
  --split test
