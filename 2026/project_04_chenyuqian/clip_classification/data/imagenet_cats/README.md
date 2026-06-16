# ImageNet Cat Images

Put ImageNet validation images from cat-related classes here if you have access to them.

Suggested classes:

- tabby cat
- tiger cat
- Persian cat
- Siamese cat
- Egyptian cat

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

Run:

```bash
python cat_alignment_analysis.py --imagenet-cat-dir data/imagenet_cats
```

If this directory contains no images, the analysis script records the ImageNet section as skipped instead of fabricating results.
