"""
Unified preprocessing module for ControlNet conditioning images.

All preprocessors accept PIL Image input and return PIL Image output,
standardized to the target resolution (default 512x512).

Supported preprocessors:
    - canny: Canny edge detection (cv2)
    - lineart: Hand-drawn lineart processing for coloring
    - lineart_anime: Anime-style lineart extraction from photos
    - scribble: Doodle/sketch cleanup for scribble-to-image
    - openpose: OpenPose skeleton extraction (MediaPipe)
    - depth: Depth map estimation (DepthAnything)
    - identity: Pass-through (no preprocessing)
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image


# ============================================================
#  Utility Functions
# ============================================================

def _ensure_numpy(img, mode="rgb"):
    """Convert PIL Image to numpy array in specified mode."""
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB" if mode == "rgb" else "L"))
    raise TypeError(f"Unsupported image type: {type(img)}")


def _ensure_pil(img, mode="RGB"):
    """Convert numpy array to PIL Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img_rgb = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
        return Image.fromarray(img_rgb)
    raise TypeError(f"Unsupported image type: {type(img)}")


def standardize_image(img, target_size=512):
    """
    Standardize image: resize keeping aspect ratio, center crop to square.
    Returns PIL RGB image at target_size x target_size.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img)
    img = img.convert("RGB")

    w, h = img.size
    # Resize so shortest side = target_size
    if h > w:
        new_h, new_w = int(target_size * h / w), target_size
    else:
        new_h, new_w = target_size, int(target_size * w / h)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img_cropped = img_resized.crop((left, top, left + target_size, top + target_size))

    return img_cropped


# ============================================================
#  Individual Preprocessors
# ============================================================

class CannyPreprocessor:
    """Canny edge detection for ControlNet Canny conditioning."""

    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image, output_size=512):
        """
        Args:
            image: PIL Image or numpy array
            output_size: target square size
        Returns:
            PIL Image: white-background black-edge conditioning image
        """
        pil_img = standardize_image(image, output_size)
        gray = np.array(pil_img.convert("L"))

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        # Invert to white background, black lines (ControlNet convention)
        edges_inv = 255 - edges
        return Image.fromarray(cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB))


class LineartPreprocessor:
    """
    Lineart preprocessing for hand-drawn / scanned lineart coloring.
    Fixes background unevenness, broken lines, noise, and unifies line width.
    Produces TWO outputs: [processed_lineart, canny_edge_map]
    for dual ControlNet injection (Lineart + Canny).
    """

    def __init__(self, line_thickness=2):
        self.line_thickness = line_thickness

    def __call__(self, image, output_size=512):
        pil_img = standardize_image(image, output_size)
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. Background correction: remove uneven illumination
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_bg)
        corrected = cv2.divide(gray, background, scale=255)

        # 2. Otsu binarization
        _, binary = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 4. Close broken lines
        kernel_med = np.ones((3, 3), np.uint8)
        binary_closed = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_med, iterations=1)

        # 5. Skeleton extraction (thinning)
        size = np.size(binary_closed)
        skel = np.zeros(binary_closed.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = binary_closed.copy()
        done = False
        while not done:
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skel = cv2.bitwise_or(skel, temp)
            working = eroded.copy()
            zeros = size - cv2.countNonZero(working)
            if zeros == size:
                done = True

        # 6. Dilate to target line thickness
        if self.line_thickness > 1:
            kernel_thick = np.ones((self.line_thickness, self.line_thickness), np.uint8)
            skel = cv2.dilate(skel, kernel_thick, iterations=1)

        # 7. Invert: white background, black lines
        final_lineart = 255 - skel
        lineart_pil = Image.fromarray(cv2.cvtColor(final_lineart, cv2.COLOR_GRAY2RGB))

        # 8. Generate Canny edge map as secondary conditioning
        canny_edges = cv2.Canny(final_lineart, threshold1=50, threshold2=150)
        canny_edges = 255 - canny_edges  # white bg, black lines
        canny_pil = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB))

        return [lineart_pil, canny_pil]


class AnimeLineartPreprocessor:
    """
    Extract anime-style clean lineart from real photos.
    Uses bilateral filtering + adaptive thresholding + skeleton extraction.
    """

    def __init__(self, line_thickness=1):
        self.line_thickness = line_thickness

    def __call__(self, image, output_size=512):
        pil_img = standardize_image(image, output_size)
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. Bilateral filter: preserve edges, remove texture
        blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 2. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=15, C=5
        )

        # 3. Morphological opening: remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Skeleton extraction
        size = np.size(binary_clean)
        skel = np.zeros(binary_clean.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = binary_clean.copy()
        done = False
        while not done:
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skel = cv2.bitwise_or(skel, temp)
            working = eroded.copy()
            zeros = size - cv2.countNonZero(working)
            if zeros == size:
                done = True

        # 5. Adjust line thickness
        if self.line_thickness > 1:
            kernel_thick = np.ones((self.line_thickness, self.line_thickness), np.uint8)
            skel = cv2.dilate(skel, kernel_thick, iterations=1)

        # 6. Invert for white background, black lines
        lineart = 255 - skel
        return Image.fromarray(cv2.cvtColor(lineart, cv2.COLOR_GRAY2RGB))


class ScribblePreprocessor:
    """
    Preprocess hand-drawn doodle/sketch for Scribble ControlNet.
    Cleans up eraser marks, stray lines, background texture, and uneven lighting.
    """

    def __init__(self, filter_strength=1):
        self.filter_strength = filter_strength

    def __call__(self, image, output_size=512):
        pil_img = standardize_image(image, output_size)
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE contrast equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # 2. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=21, C=10
        )

        # 3. Connected component analysis: remove small speckles
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = 10 * self.filter_strength
        clean_binary = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_binary[labels == i] = 255

        # 4. Remove thin guide lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        clean_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_OPEN, kernel, iterations=self.filter_strength)

        # 5. Slight dilation for line visibility
        kernel_dilate = np.ones((2, 2), np.uint8)
        clean_binary = cv2.dilate(clean_binary, kernel_dilate, iterations=1)

        # 6. Invert for white bg, black lines
        final_sketch = 255 - clean_binary
        return Image.fromarray(cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2RGB))


class DepthPreprocessor:
    """
    Depth map estimation using DepthAnything V1 (ViT-Small).
    Falls back to a simple intensity-based depth approximation if the model is unavailable.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.transform = None
        self.device = "cpu"
        self._init_model(model_path)

    def _init_model(self, model_path):
        """
        Try to load DepthAnything model from local cv_preprocess package.

        DepthAnything requires:
        1. depth_anything/ package (at cv_preprocess/depth_anything/)
        2. dinov2/ backbone (at cv_preprocess/dinov2/) — loaded via torch.hub.load('./dinov2', ...)
        3. Pretrained weights (at cv_preprocess/models/depth_anything_vits14.pth)
        """
        try:
            import torch
            import torch.nn.functional as F

            # Add cv_preprocess to path so depth_anything module is importable
            # __file__ = src/pipeline/preprocessors.py → need to go up 3 levels to project root
            cv_preprocess_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "cv_preprocess"
            )
            if cv_preprocess_dir not in sys.path:
                sys.path.insert(0, cv_preprocess_dir)

            # DepthAnything uses torch.hub.load('./dinov2', ...) with relative path.
            # Save and change working directory to cv_preprocess/ so the relative
            # './dinov2' path resolves correctly to cv_preprocess/dinov2/
            import os as _os
            _original_cwd = _os.getcwd()
            _os.chdir(cv_preprocess_dir)

            try:
                from depth_anything.dpt import DepthAnything
                from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
                from torchvision.transforms import Compose

                # Model configurations (matching the original cv_preprocess setup)
                model_configs = {
                    "vits": {
                        "encoder": "vits",
                        "features": 64,
                        "out_channels": [48, 96, 192, 384]
                    }
                }

                # Determine checkpoint path
                if model_path is None:
                    model_path = os.path.join(cv_preprocess_dir, "models", "depth_anything_vits14.pth")

                if not os.path.exists(model_path):
                    print(f"[DepthPreprocessor] Model checkpoint not found: {model_path}")
                    self.model = None
                    _os.chdir(_original_cwd)
                    return

                self.device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load model (this triggers torch.hub.load('./dinov2', ...))
                self.model = DepthAnything(model_configs["vits"])
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()

                # Setup transform pipeline
                self.transform = Compose([
                    Resize(
                        width=518, height=518,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    PrepareForNet()
                ])

                print(f"[DepthPreprocessor] DepthAnything loaded successfully (device={self.device})")
            finally:
                _os.chdir(_original_cwd)

        except Exception as e:
            print(f"[DepthPreprocessor] DepthAnything unavailable ({e}), using intensity fallback")
            self.model = None

    def __call__(self, image, output_size=512):
        """
        Args:
            image: PIL Image
            output_size: target size
        Returns:
            PIL Image: depth map as RGB image
        """
        pil_img = standardize_image(image, output_size)

        if self.model is not None and self.transform is not None:
            return self._depth_anything_infer(pil_img, output_size)
        else:
            return self._intensity_fallback(pil_img, output_size)

    def _depth_anything_infer(self, pil_img, output_size):
        """Use DepthAnything model for inference (matches original cv_preprocess/depth_preprocess.py)."""
        import torch
        import torch.nn.functional as F

        # Convert PIL to numpy [0, 1] float32 (H, W, 3) RGB
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        # Apply the same transform pipeline as the original
        transformed = self.transform({"image": img_np})
        input_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(input_tensor)
            # Interpolate back to original image size
            depth = F.interpolate(
                depth[:, None],
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )[0, 0]

        depth_np = depth.cpu().numpy()

        # Normalize to 0-255
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max > depth_min:
            depth_np = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
        depth_np = depth_np.astype(np.uint8)

        # Resize to output size
        depth_cv = cv2.resize(depth_np, (output_size, output_size), interpolation=cv2.INTER_AREA)

        # Post-processing: enhance contrast for better ControlNet conditioning
        depth_normalized = cv2.normalize(depth_cv, None, 0, 255, cv2.NORM_MINMAX)
        depth_smoothed = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
        depth_equalized = cv2.equalizeHist(depth_smoothed)

        return Image.fromarray(cv2.cvtColor(depth_equalized, cv2.COLOR_GRAY2RGB))

    def _intensity_fallback(self, pil_img, output_size):
        """
        Enhanced intensity-based depth approximation using multiple visual cues.

        Combines:
        1. Gradient magnitude (edges → depth boundaries)
        2. Inverted intensity (darker ≈ further in many photos)
        3. Texture density via local standard deviation
        4. CLAHE + Gaussian for smooth, clean depth map
        """
        gray = np.array(pil_img.convert("L")).astype(np.float32)

        # Cue 1: Gradient magnitude (captures depth discontinuities)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Cue 2: Inverted intensity (heuristic: darker regions tend to be further)
        intensity_inv = 255 - gray

        # Cue 3: Local texture density (textured regions tend to be closer)
        local_std = cv2.GaussianBlur(gray, (15, 15), 0)
        texture = cv2.Laplacian(local_std, cv2.CV_32F, ksize=5)
        texture = np.abs(texture)

        # Normalize each cue to [0, 255]
        def norm_to_255(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                return (arr - arr_min) / (arr_max - arr_min) * 255
            return np.zeros_like(arr)

        cue_grad = norm_to_255(gradient_mag)
        cue_intensity = norm_to_255(intensity_inv)
        cue_texture = norm_to_255(texture)

        # Weighted fusion (gradients provide structure, intensity provides layout)
        depth = 0.15 * cue_grad + 0.70 * cue_intensity + 0.15 * cue_texture

        # Normalize and convert
        depth = norm_to_255(depth).astype(np.uint8)

        # Post-processing for cleaner ControlNet input
        depth = cv2.GaussianBlur(depth, (7, 7), 0)
        depth = cv2.equalizeHist(depth)

        # Resize to target
        depth = cv2.resize(depth, (output_size, output_size), interpolation=cv2.INTER_AREA)

        return Image.fromarray(cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB))


class OpenPosePreprocessor:
    """
    Generate OpenPose-style skeleton map using MediaPipe Pose detection.
    Falls back to a blank image if MediaPipe is unavailable.
    """

    _pose_landmarker = None
    _face_landmarker = None

    def __init__(self, detect_face=True):
        self.detect_face = detect_face

    def _get_pose_landmarker(self):
        """Lazy-load MediaPipe PoseLandmarker."""
        if OpenPosePreprocessor._pose_landmarker is not None:
            return OpenPosePreprocessor._pose_landmarker

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp_python.vision.PoseLandmarker
            PoseLandmarkerOptions = mp_python.vision.PoseLandmarkerOptions
            RunningMode = mp_python.vision.RunningMode

            # Find model file
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cv_preprocess", "models"
            )
            model_path = os.path.join(model_dir, "pose_landmarker_full.task")

            if not os.path.exists(model_path):
                print(f"[OpenPose] Model not found: {model_path}")
                return None

            base_options = BaseOptions(model_asset_path=model_path)
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            OpenPosePreprocessor._pose_landmarker = PoseLandmarker.create_from_options(options)
            return OpenPosePreprocessor._pose_landmarker

        except Exception as e:
            print(f"[OpenPose] Failed to load MediaPipe pose: {e}")
            return None

    def _get_face_landmarker(self):
        """Lazy-load MediaPipe FaceLandmarker."""
        if OpenPosePreprocessor._face_landmarker is not None:
            return OpenPosePreprocessor._face_landmarker

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python

            FaceLandmarker = mp_python.vision.FaceLandmarker
            FaceLandmarkerOptions = mp_python.vision.FaceLandmarkerOptions
            BaseOptions = mp.tasks.BaseOptions
            RunningMode = mp_python.vision.RunningMode

            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cv_preprocess", "models"
            )
            model_path = os.path.join(model_dir, "face_landmarker.task")

            if not os.path.exists(model_path):
                print(f"[OpenPose] Face model not found: {model_path}")
                return None

            base_options = BaseOptions(model_asset_path=model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            OpenPosePreprocessor._face_landmarker = FaceLandmarker.create_from_options(options)
            return OpenPosePreprocessor._face_landmarker

        except Exception as e:
            print(f"[OpenPose] Failed to load MediaPipe face: {e}")
            return None

    def __call__(self, image, output_size=512):
        """
        Generate OpenPose skeleton visualization.
        Returns a black-background skeleton map (ControlNet standard format).
        """
        import mediapipe as mp

        pil_img = standardize_image(image, output_size)
        img_np = np.array(pil_img)

        h, w = img_np.shape[:2]
        pose_img = np.zeros((h, w, 3), dtype=np.uint8)

        pose_landmarker = self._get_pose_landmarker()

        if pose_landmarker is None:
            print("[OpenPose] Pose landmarker unavailable, returning blank map")
            return Image.fromarray(pose_img)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            result = pose_landmarker.detect(mp_image)

            if result.pose_landmarks:
                for landmarks in result.pose_landmarks:
                    # Draw keypoints
                    for lm in landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(pose_img, (x, y), 4, (255, 255, 255), -1)

                    # Draw connections (ControlNet standard 33-keypoint skeleton)
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6),
                        (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17),
                        (15, 19), (15, 21), (12, 14), (14, 16), (16, 18), (16, 20),
                        (16, 22), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                        (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
                    ]
                    for si, ei in connections:
                        if si < len(landmarks) and ei < len(landmarks):
                            sx, sy = int(landmarks[si].x * w), int(landmarks[si].y * h)
                            ex, ey = int(landmarks[ei].x * w), int(landmarks[ei].y * h)
                            cv2.line(pose_img, (sx, sy), (ex, ey), (255, 255, 255), 2)

            # Draw face landmarks if enabled
            if self.detect_face:
                face_landmarker = self._get_face_landmarker()
                if face_landmarker is not None:
                    face_result = face_landmarker.detect(mp_image)
                    if face_result.face_landmarks:
                        for face_landmarks in face_result.face_landmarks:
                            for lm in face_landmarks:
                                x, y = int(lm.x * w), int(lm.y * h)
                                cv2.circle(pose_img, (x, y), 2, (255, 255, 255), -1)

        except Exception as e:
            print(f"[OpenPose] Detection error: {e}")

        return Image.fromarray(pose_img)


class IdentityPreprocessor:
    """Pass-through preprocessor (no processing)."""
    def __call__(self, image, output_size=512):
        return standardize_image(image, output_size)


# ============================================================
#  Preprocessor Registry
# ============================================================

class PreprocessorRegistry:
    """Central registry for accessing preprocessors by name."""

    _instances = {}

    @classmethod
    def get(cls, name, **kwargs):
        """
        Get or create a preprocessor instance.

        Args:
            name: One of 'canny', 'lineart', 'lineart_anime', 'scribble',
                  'openpose', 'depth', 'identity'
            **kwargs: Passed to preprocessor constructor.

        Returns:
            A callable preprocessor instance.
        """
        if name not in cls._instances:
            preprocessors = {
                "canny": lambda: CannyPreprocessor(**kwargs) if kwargs else CannyPreprocessor(),
                "lineart": lambda: LineartPreprocessor(**kwargs) if kwargs else LineartPreprocessor(),
                "lineart_anime": lambda: AnimeLineartPreprocessor(**kwargs) if kwargs else AnimeLineartPreprocessor(),
                "scribble": lambda: ScribblePreprocessor(**kwargs) if kwargs else ScribblePreprocessor(),
                "depth": lambda: DepthPreprocessor(**kwargs) if kwargs else DepthPreprocessor(),
                "openpose": lambda: OpenPosePreprocessor(**kwargs) if kwargs else OpenPosePreprocessor(),
                "identity": lambda: IdentityPreprocessor(),
            }
            if name not in preprocessors:
                raise ValueError(f"Unknown preprocessor: {name}. Available: {list(preprocessors.keys())}")
            cls._instances[name] = preprocessors[name]()

        return cls._instances[name]

    @classmethod
    def get_multi(cls, names, **kwargs):
        """Get multiple preprocessors at once. Returns list."""
        return [cls.get(n, **kwargs) for n in names]

    @classmethod
    def clear_cache(cls):
        """Clear cached preprocessor instances."""
        cls._instances.clear()

    @classmethod
    def list_all(cls):
        return ["canny", "lineart", "lineart_anime", "scribble", "depth", "openpose", "identity"]
