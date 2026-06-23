# Test Examples

This folder contains three lightweight ImageNet-S validation images used by `python -m src.main` and `python -m src.infer` for the course-required sample inference.

The examples correspond to classes used by the current 10-class baseline:

- `imagenet_demo_01.jpg`: goldfish (`n01443537`)
- `imagenet_demo_02.jpg`: airliner (`n02690373`)
- `imagenet_demo_03.jpg`: dog (`n02104029`)

They are small demonstration inputs, not the full ImageNet-S dataset. Expected output files are generated under `results/demo_inference_10cls/` or the output directory passed on the command line:

- `pred_mask.png`: raw predicted label ids
- `pred_color.png`: colorized predicted segmentation map
- `overlay.png`: segmentation overlay on the input image
- `pred_labels.txt`: unique predicted label ids found in the image
