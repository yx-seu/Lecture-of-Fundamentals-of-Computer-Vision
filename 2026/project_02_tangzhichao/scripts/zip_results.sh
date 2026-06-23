#!/usr/bin/env bash
set -e
zip -r imagenet_s_segmentation_results.zip outputs/imagenet_s_animals_10cls_trainval_holdout_ade results README.md references.md requirements.txt configs src demos data/test_examples data/dataset_info.txt
