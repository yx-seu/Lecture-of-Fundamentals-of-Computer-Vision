#!/usr/bin/env bash
set -e
python -m src.train --config configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml
