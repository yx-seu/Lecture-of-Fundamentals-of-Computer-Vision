# References & Third-Party Code Attribution

## Model Weights

### Stable Diffusion 1.5
- **Source**: [runwayml/stable-diffusion-v1-5](https://hf-mirror.com/stable-diffusion-v1-5/stable-diffusion-v1-5)
- **License**: CreativeML Open RAIL-M License
- **Citation**:
  ```bibtex
  @misc{rombach2022highresolution,
    title={High-Resolution Image Synthesis with Latent Diffusion Models},
    author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
    year={2022},
    eprint={2112.10752},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

### ControlNet
- **Source**: [lllyasviel/ControlNet-v1-1](https://hf-mirror.com/lllyasviel/models)
- **Models used**: control_v11p_sd15_canny, control_v11f1p_sd15_depth, control_v11p_sd15_lineart, control_v11p_sd15_scribble, control_v11p_sd15_openpose, control_v11p_sd15_lineart_anime
- **License**: OpenRAIL License
- **Citation**:
  ```bibtex
  @misc{zhang2023adding,
    title={Adding Conditional Control to Text-to-Image Diffusion Models},
    author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
    year={2023},
    eprint={2302.05543},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

### DepthAnything V1
- **Source**: [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- **Checkpoint**: `depth_anything_vits14.pth` (ViT-Small, 24.8M parameters)
- **License**: Apache 2.0
- **Citation**:
  ```bibtex
  @misc{yang2024depth,
    title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
    author={Lihe Yang and Bingyi Kang and Zilong Huang and Xiaogang Xu and Jiashi Feng and Hengshuang Zhao},
    year={2024},
    eprint={2401.10891},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

### DINOv2
- **Source**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **Used as**: Backbone encoder for DepthAnything
- **License**: Apache 2.0
- **Citation**:
  ```bibtex
  @misc{oquab2023dinov2,
    title={DINOv2: Learning Robust Visual Features without Supervision},
    author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
    year={2023},
    eprint={2304.07193},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

## Software Libraries

### Diffusers
- **Source**: [huggingface/diffusers](https://github.com/huggingface/diffusers) v0.25.0
- **License**: Apache 2.0
- **Usage**: `StableDiffusionControlNetPipeline`, `ControlNetModel`, scheduler classes
- **Citation**:
  ```bibtex
  @misc{von-platen-etal-2022-diffusers,
    author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
    title = {Diffusers: State-of-the-art diffusion models},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huggingface/diffusers}}
  }
  ```

### MediaPipe
- **Source**: [google/mediapipe](https://github.com/google/mediapipe) v0.10.35
- **License**: Apache 2.0
- **Usage**: `PoseLandmarker`, `FaceLandmarker` for OpenPose-style skeleton generation

### Gradio
- **Source**: [gradio-app/gradio](https://github.com/gradio-app/gradio) v4.44.1
- **License**: Apache 2.0
- **Usage**: Web UI framework for the interactive interface

### OpenCV
- **Source**: [opencv/opencv-python](https://github.com/opencv/opencv-python) v4.9.0
- **License**: Apache 2.0
- **Usage**: Canny edge detection, morphological operations, skeleton extraction, color space conversions

## Preprocessing Algorithm References

### Lineart / AnimeLineart Preprocessing
The hand-crafted lineart extraction pipeline (background correction → Otsu binarization → morphological cleaning → skeleton extraction) was developed specifically for this project, drawing on standard computer vision techniques:
- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
- Zhang, T.Y. & Suen, C.Y. (1984). "A Fast Parallel Algorithm for Thinning Digital Patterns." *Communications of the ACM*, 27(3), 236–239.

### Scribble Preprocessing
The connected-component-based noise filtering approach follows:
- Samet, H. & Tamminen, M. (1988). "Efficient Component Labeling of Images of Arbitrary Dimension." *IEEE TPAMI*, 10(4), 579–586.

### Depth Fallback
The multi-cue intensity-based depth approximation combines:
- Gradient magnitude (Sobel operator)
- Intensity inversion (dark-is-deep heuristic)
- Local texture density (Laplacian variance)

## Known Compatibility Issues & Resolutions

1. **controlnet_aux mediapipe incompatibility**: The `controlnet_aux` package (v0.0.10) uses the deprecated `mediapipe.solutions` API, which was removed in `mediapipe>=0.10`. Resolution: all preprocessors re-implemented using OpenCV and the new `mediapipe.tasks` API directly.

2. **Starlette 1.1.0 TemplateResponse signature change**: `TemplateResponse(self, name, context)` → `TemplateResponse(self, request, name, context)`. Resolution: downgraded Starlette to 0.38.6 (compatible with Gradio 4.44.1).

3. **Gradio gradio_client boolean JSON schema bug**: `_json_schema_to_python_type()` crashes on `"additionalProperties": true`. Resolution: runtime monkey-patch applied in `main.py` before Gradio imports.

---

*All third-party code and model weights are used in accordance with their respective licenses. The custom preprocessing pipelines and scenario orchestration code are original work for this project.*
