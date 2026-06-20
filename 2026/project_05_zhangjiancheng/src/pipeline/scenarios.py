"""
Scenario-specific pipelines for the 4 target use cases.

Each scenario provides:
    - Optimized default parameters
    - Curated prompt templates
    - Integrated preprocessing → generation → output workflow
    - Style selection and parameter tuning

Scenarios:
    1. lineart_coloring     - Hand-drawn lineart auto-coloring (Lineart + Canny)
    2. sketch_to_realistic  - Doodle/sketch to photorealistic image (Scribble)
    3. photo_to_anime       - Real photo to anime style (AnimeLineart + OpenPose + Depth)
    4. old_photo_restore    - Old/damaged photo restoration (Canny + Depth)
"""

import os
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
from PIL import Image

from .inference import ControlNetInference
from .preprocessors import PreprocessorRegistry, standardize_image

logger = logging.getLogger(__name__)


# ============================================================
#  Data Classes
# ============================================================

class Scenario(Enum):
    LINERART_COLORING = "lineart_coloring"
    SKETCH_TO_REALISTIC = "sketch_to_realistic"
    PHOTO_TO_ANIME = "photo_to_anime"
    OLD_PHOTO_RESTORE = "old_photo_restore"


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario pipeline."""
    name: str
    description: str
    controlnet_names: List[str]           # Which ControlNets to use
    preprocessor_names: List[str]         # Which preprocessors to run
    prompt_templates: Dict[str, str]      # Style -> prompt mapping
    negative_prompt: str = ""
    default_steps: int = 25
    default_cfg: float = 7.5
    default_scales: List[float] = field(default_factory=list)
    default_scheduler: str = "dpm++"
    default_width: int = 512
    default_height: int = 512

    def __post_init__(self):
        if not self.default_scales:
            self.default_scales = [0.85] * len(self.controlnet_names)


# ============================================================
#  Scenario Pipeline
# ============================================================

class ScenarioPipeline:
    """
    Orchestrates the full workflow for each scenario:
        Input Image → Preprocessing → ControlNet Inference → Output Image

    Usage:
        sp = ScenarioPipeline(engine)
        result = sp.run_lineart_coloring(
            image=lineart_image,
            style="vivid_anime",
            steps=25,
            controlnet_scales=[0.9, 0.7],
        )
    """

    def __init__(
        self,
        engine: ControlNetInference,
        sd_models_dir: str,
    ):
        """
        Args:
            engine: ControlNetInference instance with SD model loaded.
            sd_models_dir: Root directory containing all ControlNet model folders.
        """
        self.engine = engine
        self.models_dir = sd_models_dir

        # Map ControlNet short names to actual directory names
        # (handles naming inconsistencies like 'lineart_anime' → 'control-animelineart')
        self._controlnet_dir_map = {
            "lineart": "control-lineart",
            "canny": "control-canny",
            "depth": "control-depth",
            "openpose": "control-openpose",
            "scribble": "control-scribble",
            "lineart_anime": "control-animelineart",
        }

        # Define all 4 scenarios
        self._scenarios: Dict[str, ScenarioConfig] = {
            Scenario.LINERART_COLORING.value: ScenarioConfig(
                name="线稿自动上色",
                description="输入手绘线稿，一键自动上色，支持多种风格",
                controlnet_names=["lineart", "canny"],
                preprocessor_names=["lineart"],  # Returns [lineart, canny]
                prompt_templates={
                    "vivid_anime": "anime illustration, vibrant colors, cel shaded, "
                                   "clean coloring, high saturation, studio ghibli inspired",
                    "watercolor": "watercolor painting style, soft colors, artistic, "
                                  "delicate brush strokes, ethereal",
                    "oil_painting": "oil painting, rich textures, classical art style, "
                                    "detailed brushwork, canvas texture",
                    "flat_color": "flat color illustration, minimalist, vector art style, "
                                  "solid colors, bold outlines, pop art",
                    "digital_art": "digital painting, concept art, professional illustration, "
                                   "smooth gradients, detailed",
                },
                negative_prompt=(
                    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                    "extra limbs, missing details, oversaturated, color bleeding, "
                    "muddy colors, gray, desaturated, wrong colors"
                ),
                default_steps=25,
                default_cfg=8.0,
                default_scales=[0.9, 0.65],
                default_scheduler="dpm++",
            ),

            Scenario.SKETCH_TO_REALISTIC.value: ScenarioConfig(
                name="草图转写实效果图",
                description="输入涂鸦草图，生成建筑/产品写实效果图",
                controlnet_names=["scribble"],
                preprocessor_names=["scribble"],
                prompt_templates={
                    "architecture": "architectural rendering, photorealistic building exterior, "
                                    "professional architecture photography, natural lighting, "
                                    "ultra detailed, 8k, glass and concrete, modern design",
                    "product": "product photography, studio lighting, white background, "
                               "commercial photography, highly detailed, 8k, professional, "
                               "elegant design, luxury item",
                    "landscape": "photorealistic landscape, nature photography, breathtaking view, "
                                 "national geographic, ultra detailed, natural lighting, 8k",
                    "interior": "interior design rendering, photorealistic room, natural lighting, "
                                "architectural digest, high end furniture, ultra detailed, 8k",
                },
                negative_prompt=(
                    "blurry, low quality, distorted, cartoon, illustration, painting, "
                    "sketch, drawing, deformed, ugly, bad anatomy"
                ),
                default_steps=30,
                default_cfg=9.0,
                default_scales=[0.95],
                default_scheduler="dpm++",
            ),

            Scenario.PHOTO_TO_ANIME.value: ScenarioConfig(
                name="真人/实景转动漫风",
                description="输入真人照片或实景，转换为动漫风格",
                controlnet_names=["lineart_anime", "openpose", "depth"],
                preprocessor_names=["lineart_anime", "openpose", "depth"],
                prompt_templates={
                    "anime_film": "anime movie screenshot, studio ghibli style, makoto shinkai, "
                                  "beautiful scenery, high quality animation, cinematic lighting",
                    "manga": "manga illustration style, black and white with subtle color, "
                             "crosshatching, detailed linework, seinen manga",
                    "anime_portrait": "anime portrait, character design, detailed face, "
                                      "vibrant eyes, soft shading, high quality anime art",
                    "chibi": "chibi style, cute, adorable, super deformed, kawaii, "
                             "colorful, simple, cheerful",
                },
                negative_prompt=(
                    "photorealistic, realistic, 3d render, photo, photograph, "
                    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                    "extra fingers, mutated hands, poorly drawn face"
                ),
                default_steps=30,
                default_cfg=8.5,
                default_scales=[0.8, 0.6, 0.7],
                default_scheduler="dpm++",
            ),

            Scenario.OLD_PHOTO_RESTORE.value: ScenarioConfig(
                name="老照片修复",
                description="修复破损老照片，增强细节，还原真实色彩",
                controlnet_names=["canny", "depth"],
                preprocessor_names=["canny", "depth"],
                prompt_templates={
                    "restore_bw": "black and white photograph, restored vintage photo, sharp, "
                                  "clean, high contrast, professional restoration, detailed",
                    "restore_color": "color photograph, restored vintage photo, natural colors, "
                                     "clean, sharp details, professional restoration, lifelike",
                    "enhance": "high quality photograph, restored, enhanced, sharp details, "
                               "professional photo restoration, clean, clear",
                    "colorize": "colorized vintage photograph, natural skin tones, realistic colors, "
                                "beautifully restored, professional colorization, detailed",
                },
                negative_prompt=(
                    "blurry, low quality, distorted, damaged, scratches, noise, "
                    "artifacts, deformed, ugly, oversaturated, unnatural colors, "
                    "cartoon, painting, illustration"
                ),
                default_steps=25,
                default_cfg=7.5,
                default_scales=[0.85, 0.75],
                default_scheduler="ddim",
            ),
        }

    # ================================================================
    #  High-level API: Run a scenario
    # ================================================================

    def run(
        self,
        scenario: str,
        image: Image.Image,
        style: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        controlnet_scales: Optional[List[float]] = None,
        scheduler: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 512,
        height: int = 512,
    ) -> Dict:
        """
        Run a full scenario pipeline end-to-end.

        Args:
            scenario: Scenario name (use Scenario enum values or Chinese labels won't match directly).
                      Valid: 'lineart_coloring', 'sketch_to_realistic', 'photo_to_anime', 'old_photo_restore'
            image: Input image (PIL Image).
            style: Style key for prompt template (e.g., 'vivid_anime', 'architecture').
            prompt: Override prompt (if provided, style template is ignored).
            negative_prompt: Override negative prompt.
            steps: Number of inference steps.
            cfg: CFG guidance scale.
            controlnet_scales: Per-ControlNet conditioning scales.
            scheduler: Scheduler name.
            seed: Random seed.
            width: Output image width.
            height: Output image height.

        Returns:
            Dict with keys:
                - 'output_image': PIL.Image generated output
                - 'control_images': List[PIL.Image] conditioning images
                - 'preprocessed': List[PIL.Image] preprocessed inputs
                - 'seed': int
                - 'time': float
                - 'config': ScenarioConfig used
        """
        if scenario not in self._scenarios:
            raise ValueError(
                f"Unknown scenario: '{scenario}'. "
                f"Available: {list(self._scenarios.keys())}"
            )

        config = self._scenarios[scenario]
        logger.info(f"Running scenario: {config.name} ({scenario})")

        # ---- Step 1: Preprocessing ----
        control_images = self._run_preprocessing(
            config=config,
            image=image,
            width=width,
            height=height,
        )
        logger.info(f"Preprocessing complete: {len(control_images)} control image(s)")

        # ---- Step 2: Determine ControlNets and build pipeline ----
        self.engine.load_controlnets({
            name: os.path.join(self.models_dir, self._controlnet_dir_map[name])
            for name in config.controlnet_names
        })
        self.engine.build_pipeline(controlnet_names=config.controlnet_names)

        # ---- Step 3: Set parameters ----
        if scheduler is None:
            scheduler = config.default_scheduler
        self.engine.set_scheduler(scheduler)

        if steps is None:
            steps = config.default_steps
        if cfg is None:
            cfg = config.default_cfg
        if controlnet_scales is None:
            controlnet_scales = config.default_scales

        # Resolve prompt
        if prompt is None:
            if style and style in config.prompt_templates:
                prompt = config.prompt_templates[style]
            else:
                # Use first template as default
                prompt = list(config.prompt_templates.values())[0]

        if negative_prompt is None:
            negative_prompt = config.negative_prompt

        # ---- Step 4: Generate ----
        result = self.engine.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_images=control_images[0] if len(control_images) == 1 else control_images,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=controlnet_scales[0]
                if len(controlnet_scales) == 1 else controlnet_scales,
            height=height,
            width=width,
            seed=seed,
        )

        output_image = result["images"][0]

        logger.info(f"Scenario '{scenario}' complete: seed={result['seed']}, time={result['time']:.1f}s")

        return {
            "output_image": output_image,
            "control_images": control_images,
            "seed": result["seed"],
            "time": result["time"],
            "config": config,
            "prompt_used": prompt,
            "negative_prompt_used": negative_prompt,
        }

    # ================================================================
    #  Convenience methods for each scenario
    # ================================================================

    def run_lineart_coloring(
        self,
        image: Image.Image,
        style: str = "vivid_anime",
        **kwargs,
    ) -> Dict:
        """Hand-drawn lineart auto-coloring."""
        return self.run(
            scenario=Scenario.LINERART_COLORING.value,
            image=image,
            style=style,
            **kwargs,
        )

    def run_sketch_to_realistic(
        self,
        image: Image.Image,
        style: str = "architecture",
        **kwargs,
    ) -> Dict:
        """Doodle/sketch to photorealistic rendering."""
        return self.run(
            scenario=Scenario.SKETCH_TO_REALISTIC.value,
            image=image,
            style=style,
            **kwargs,
        )

    def run_photo_to_anime(
        self,
        image: Image.Image,
        style: str = "anime_film",
        **kwargs,
    ) -> Dict:
        """Real photo to anime style conversion."""
        return self.run(
            scenario=Scenario.PHOTO_TO_ANIME.value,
            image=image,
            style=style,
            **kwargs,
        )

    def run_old_photo_restore(
        self,
        image: Image.Image,
        style: str = "restore_bw",
        **kwargs,
    ) -> Dict:
        """Old/damaged photo restoration."""
        return self.run(
            scenario=Scenario.OLD_PHOTO_RESTORE.value,
            image=image,
            style=style,
            **kwargs,
        )

    # ================================================================
    #  Preprocessing Orchestration
    # ================================================================

    def _run_preprocessing(
        self,
        config: ScenarioConfig,
        image: Image.Image,
        width: int = 512,
        height: int = 512,
    ) -> List[Image.Image]:
        """
        Run preprocessing for a scenario.

        Some preprocessors return multiple images (e.g., LineartPreprocessor
        returns [lineart, canny] for dual ControlNet use). This method handles
        flattening to match the expected ControlNet count.
        """
        control_images = []
        img_size = max(width, height)

        for preproc_name in config.preprocessor_names:
            preprocessor = PreprocessorRegistry.get(preproc_name)
            result = preprocessor(image, output_size=img_size)

            if isinstance(result, list):
                control_images.extend(result)
            else:
                control_images.append(result)

        # Ensure we have the right number of control images
        # If a preprocessor returned multiple images, we may have extras;
        # if too few, duplicate the last one
        expected_count = len(config.controlnet_names)

        if len(control_images) > expected_count:
            logger.warning(
                f"Got {len(control_images)} control images for {expected_count} ControlNets; "
                f"using first {expected_count}"
            )
            control_images = control_images[:expected_count]
        elif len(control_images) < expected_count:
            logger.warning(
                f"Got {len(control_images)} control images for {expected_count} ControlNets; "
                f"padding with duplicates"
            )
            while len(control_images) < expected_count:
                control_images.append(control_images[-1])

        return control_images

    # ================================================================
    #  Utility
    # ================================================================

    def get_styles(self, scenario: str) -> List[str]:
        """Get available style options for a scenario."""
        if scenario not in self._scenarios:
            return []
        return list(self._scenarios[scenario].prompt_templates.keys())

    def get_config(self, scenario: str) -> ScenarioConfig:
        """Get the full configuration for a scenario."""
        if scenario not in self._scenarios:
            raise ValueError(f"Unknown scenario: '{scenario}'")
        return self._scenarios[scenario]

    def get_all_scenarios(self) -> List[Dict]:
        """Get summary of all available scenarios."""
        return [
            {
                "id": key,
                "name": cfg.name,
                "description": cfg.description,
                "styles": list(cfg.prompt_templates.keys()),
                "controlnets": cfg.controlnet_names,
            }
            for key, cfg in self._scenarios.items()
        ]
