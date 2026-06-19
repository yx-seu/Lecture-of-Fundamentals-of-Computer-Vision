"""
Core inference engine for Stable Diffusion + ControlNet controllable generation.

Supports:
    - Single or multiple ControlNet injection
    - Multiple samplers: DDIM, Euler, DPM++
    - Full parameter control: steps, CFG, ControlNet weights, seed, image size
    - Memory optimization for consumer GPUs (8GB VRAM)
"""

import os
import time
import logging
from typing import List, Optional, Union, Dict, Tuple

import torch
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)

logger = logging.getLogger(__name__)


# ============================================================
#  Scheduler Registry
# ============================================================

SCHEDULER_MAP = {
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "dpm++": DPMSolverMultistepScheduler,  # alias
    "dpmsolver": DPMSolverMultistepScheduler,
    "pndm": PNDMScheduler,
}

SCHEDULER_ALIASES = {
    "ddim": "DDIM",
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "dpm": "DPM++",
    "dpm++": "DPM++",
    "dpmsolver": "DPM++",
    "pndm": "PNDM",
}


# ============================================================
#  Main Inference Class
# ============================================================

class ControlNetInference:
    """
    Wraps StableDiffusionControlNetPipeline with full parameter support.

    Usage:
        engine = ControlNetInference(
            sd_model_path="models/stable-diffusion-v1-5",
            device="cuda",
            torch_dtype=torch.float16,
        )
        engine.load_controlnets({
            "canny": "models/control-canny",
            "depth": "models/control-depth",
        })
        engine.set_scheduler("dpm++")

        results = engine.generate(
            prompt="a beautiful landscape, masterpiece",
            control_images=[canny_img, depth_img],
            controlnet_scales=[0.8, 0.6],
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=42,
        )
    """

    def __init__(
        self,
        sd_model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        local_files_only: bool = True,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        disable_safety_checker: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            sd_model_path: Path to Stable Diffusion 1.5 model directory.
            device: Device to run on ('cuda' or 'cpu').
            torch_dtype: Torch data type (torch.float16 recommended for GPU).
            local_files_only: Only use local files (no HF Hub download).
            enable_attention_slicing: Reduce VRAM usage (~20% less).
            enable_vae_slicing: Reduce VRAM during VAE decode.
            disable_safety_checker: Disable NSFW filter for speed.
        """
        self.sd_model_path = sd_model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only

        self.controlnet_models: Dict[str, ControlNetModel] = {}
        self.controlnet_paths: Dict[str, str] = {}
        self.pipe: Optional[StableDiffusionControlNetPipeline] = None
        self.current_scheduler: str = "pndm"  # SD 1.5 default

        self._enable_attention_slicing = enable_attention_slicing
        self._enable_vae_slicing = enable_vae_slicing
        self._disable_safety_checker = disable_safety_checker

        logger.info(f"ControlNetInference initialized (device={device}, dtype={torch_dtype})")

    # ================================================================
    #  Model Loading
    # ================================================================

    def load_controlnet(self, name: str, model_path: str) -> ControlNetModel:
        """
        Load a single ControlNet model.

        Args:
            name: Human-readable name (e.g., 'canny', 'lineart').
            model_path: Path to ControlNet model directory.

        Returns:
            The loaded ControlNetModel.
        """
        if name in self.controlnet_models:
            logger.info(f"ControlNet '{name}' already loaded, reusing.")
            return self.controlnet_models[name]

        logger.info(f"Loading ControlNet '{name}' from {model_path}...")
        t0 = time.time()

        controlnet = ControlNetModel.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
        )

        self.controlnet_models[name] = controlnet
        self.controlnet_paths[name] = model_path

        logger.info(f"ControlNet '{name}' loaded in {time.time() - t0:.1f}s")
        return controlnet

    def load_controlnets(self, model_map: Dict[str, str]):
        """
        Load multiple ControlNet models at once.

        Args:
            model_map: Dict mapping name -> path, e.g.
                       {'canny': 'models/control-canny', 'depth': 'models/control-depth'}
        """
        for name, path in model_map.items():
            self.load_controlnet(name, path)

    def build_pipeline(self, controlnet_names: Optional[List[str]] = None):
        """
        Build (or rebuild) the StableDiffusionControlNetPipeline with the
        specified ControlNet(s).

        Args:
            controlnet_names: List of ControlNet names to use.
                If None, uses all loaded ControlNets.
                Pass a single-element list for single ControlNet.
                Pass multiple for multi-ControlNet injection.
        """
        if controlnet_names is None:
            controlnet_names = list(self.controlnet_models.keys())

        if not controlnet_names:
            raise ValueError("No ControlNet models specified. Call load_controlnets() first.")

        # Collect ControlNet models
        controlnets = []
        for name in controlnet_names:
            if name not in self.controlnet_models:
                raise KeyError(f"ControlNet '{name}' not loaded. Available: {list(self.controlnet_models.keys())}")
            controlnets.append(self.controlnet_models[name])

        logger.info(f"Building pipeline with ControlNets: {controlnet_names}")

        # For a single ControlNet, pass it directly (not in a list)
        # diffusers handles both single and list
        controlnet_arg = controlnets[0] if len(controlnets) == 1 else controlnets

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model_path,
            controlnet=controlnet_arg,
            torch_dtype=self.torch_dtype,
            local_files_only=self.local_files_only,
            safety_checker=None if self._disable_safety_checker else None,
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Memory optimizations
        if self._enable_attention_slicing:
            self.pipe.enable_attention_slicing()
            logger.info("Attention slicing enabled")

        if self._enable_vae_slicing:
            try:
                self.pipe.enable_vae_slicing()
                logger.info("VAE slicing enabled")
            except Exception:
                pass  # Not all pipelines support this

        # Disable safety checker
        if self._disable_safety_checker:
            self.pipe.safety_checker = None
            logger.info("Safety checker disabled")

        # Apply the current scheduler
        self._apply_scheduler(self.current_scheduler)

        logger.info(f"Pipeline built successfully (device={self.device})")
        return self.pipe

    # ================================================================
    #  Scheduler Management
    # ================================================================

    def set_scheduler(self, scheduler_name: str):
        """
        Set the noise scheduler.

        Args:
            scheduler_name: One of 'ddim', 'euler', 'euler_ancestral', 'dpm++', 'pndm'.
        """
        scheduler_name = scheduler_name.lower()
        if scheduler_name not in SCHEDULER_MAP:
            raise ValueError(
                f"Unknown scheduler '{scheduler_name}'. "
                f"Available: {list(SCHEDULER_MAP.keys())}"
            )

        self.current_scheduler = scheduler_name

        if self.pipe is not None:
            self._apply_scheduler(scheduler_name)

    def _apply_scheduler(self, scheduler_name: str):
        """Replace the pipeline scheduler with a new one of the given type."""
        scheduler_cls = SCHEDULER_MAP[scheduler_name]
        # Create a new scheduler from the config of the current one
        scheduler_config = self.pipe.scheduler.config
        self.pipe.scheduler = scheduler_cls.from_config(scheduler_config)
        logger.info(f"Scheduler set to: {SCHEDULER_ALIASES[scheduler_name]} ({scheduler_cls.__name__})")

    def get_available_schedulers(self) -> List[str]:
        """Return list of available scheduler names."""
        return list(SCHEDULER_MAP.keys())

    # ================================================================
    #  Memory Management
    # ================================================================

    def print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3
            logger.info(f"GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def offload_to_cpu(self):
        """Move pipeline to CPU to free GPU memory."""
        if self.pipe is not None:
            self.pipe = self.pipe.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Pipeline offloaded to CPU")

    def to_device(self, device: Optional[str] = None):
        """Move pipeline to the specified device."""
        target = device or self.device
        if self.pipe is not None:
            self.pipe = self.pipe.to(target)
            logger.info(f"Pipeline moved to {target}")

    # ================================================================
    #  Generation
    # ================================================================

    def generate(
        self,
        prompt: str,
        control_images: Union[Image.Image, List[Image.Image]],
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        output_type: str = "pil",
    ) -> Dict[str, object]:
        """
        Generate an image using ControlNet-conditioned Stable Diffusion.

        Args:
            prompt: Text prompt guiding the generation.
            negative_prompt: Negative text prompt (what to avoid).
            control_images: ControlNet conditioning image(s). Single PIL Image
                            for single ControlNet, list of PIL Images for multi.
            num_inference_steps: Number of denoising steps (20-50 typical).
            guidance_scale: Classifier-free guidance scale (7.5 typical, 1-20).
            controlnet_conditioning_scale: ControlNet influence weight(s).
                - 0.0 = no ControlNet influence (pure SD)
                - 1.0 = full ControlNet influence
                - List[float] for multi-ControlNet (one per ControlNet)
            height: Output image height (default 512 for SD 1.5).
            width: Output image width (default 512 for SD 1.5).
            seed: Random seed for reproducibility (None = random).
            guess_mode: ControlNet guess mode (trades spatial accuracy for diversity).
            control_guidance_start: When ControlNet guidance starts (0.0 = from beginning).
            control_guidance_end: When ControlNet guidance ends (1.0 = until end).
            output_type: 'pil' for PIL Image, 'np' for numpy array.

        Returns:
            Dict with keys:
                - 'images': List[PIL.Image] generated images
                - 'seed': int seed used
                - 'time': float generation time (seconds)
                - 'scheduler': str scheduler name
                - 'n_steps': int inference steps used
        """
        if self.pipe is None:
            self.build_pipeline()

        # Handle seed
        if seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Log generation parameters
        n_controlnets = len(self.controlnet_models)
        scales_str = controlnet_conditioning_scale if isinstance(controlnet_conditioning_scale, list) else [controlnet_conditioning_scale]
        logger.info(
            f"Generating: steps={num_inference_steps}, CFG={guidance_scale}, "
            f"ControlNet scales={scales_str}, seed={seed}, "
            f"scheduler={SCHEDULER_ALIASES.get(self.current_scheduler, self.current_scheduler)}"
        )

        t0 = time.time()

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=control_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                height=height,
                width=width,
                generator=generator,
                guess_mode=guess_mode,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                output_type=output_type,
            )

        elapsed = time.time() - t0
        logger.info(f"Generation complete in {elapsed:.1f}s ({elapsed / num_inference_steps:.2f}s/step)")

        return {
            "images": result.images,
            "seed": seed,
            "time": elapsed,
            "scheduler": self.current_scheduler,
            "n_steps": num_inference_steps,
        }

    # ================================================================
    #  Convenience Methods
    # ================================================================

    def generate_single(
        self,
        prompt: str,
        control_images: Union[Image.Image, List[Image.Image]],
        **kwargs,
    ) -> Image.Image:
        """Generate and return a single PIL Image."""
        result = self.generate(prompt=prompt, control_images=control_images, **kwargs)
        return result["images"][0]

    def generate_with_prompt_template(
        self,
        prompt_template: str,
        style_prompt: str = "",
        quality_prompt: str = "",
        negative_template: str = "",
        control_images: Union[Image.Image, List[Image.Image]] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Generate using a prompt template with style/quality modifiers.

        Args:
            prompt_template: Base prompt describing the content.
            style_prompt: Style modifier (e.g., 'anime style', 'photorealistic').
            quality_prompt: Quality booster (e.g., 'masterpiece, best quality').
            negative_template: Negative prompt template.
            control_images: Conditioning image(s).
            **kwargs: Passed to generate().
        """
        # Build full prompt
        parts = []
        if quality_prompt:
            parts.append(quality_prompt)
        if prompt_template:
            parts.append(prompt_template)
        if style_prompt:
            parts.append(style_prompt)
        full_prompt = ", ".join(parts)

        result = self.generate(
            prompt=full_prompt,
            negative_prompt=negative_template,
            control_images=control_images,
            **kwargs,
        )
        return result["images"][0]
