"""
Gradio web interface for SD + ControlNet controllable image generation.

Features:
    - Image upload (drag & drop / click)
    - Scenario & style selection
    - Parameter sliders (ControlNet weights, steps, CFG, seed)
    - Before/after image comparison
    - Error handling with user-friendly messages
    - Progress indication
"""

import os
import sys
import time
import logging
import traceback
from typing import Optional, List, Dict

import gradio as gr
import torch
from PIL import Image

# Patch applied in main.py before all imports

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.inference import ControlNetInference
from pipeline.scenarios import ScenarioPipeline, Scenario

logger = logging.getLogger(__name__)

# ============================================================
#  Constants
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SD_MODEL_PATH = os.path.join(MODELS_DIR, "stable-diffusion-v1-5")

SCENARIO_CHOICES = [
    ("🎨 线稿自动上色", "lineart_coloring"),
    ("🏗️ 草图转写实效果图", "sketch_to_realistic"),
    ("🌸 真人/实景转动漫风", "photo_to_anime"),
    ("📷 老照片修复", "old_photo_restore"),
]

# Style options per scenario
STYLE_CHOICES = {
    "lineart_coloring": [
        ("鲜艳动漫风", "vivid_anime"),
        ("水彩画风", "watercolor"),
        ("油画风格", "oil_painting"),
        ("扁平色块", "flat_color"),
        ("数字绘画", "digital_art"),
    ],
    "sketch_to_realistic": [
        ("建筑效果图", "architecture"),
        ("产品摄影", "product"),
        ("自然风光", "landscape"),
        ("室内设计", "interior"),
    ],
    "photo_to_anime": [
        ("动漫电影风", "anime_film"),
        ("漫画风格", "manga"),
        ("动漫肖像", "anime_portrait"),
        ("Q版可爱风", "chibi"),
    ],
    "old_photo_restore": [
        ("黑白修复", "restore_bw"),
        ("彩色修复", "restore_color"),
        ("细节增强", "enhance"),
        ("智能上色", "colorize"),
    ],
}

SCHEDULER_CHOICES = [
    ("DPM++ (推荐)", "dpm++"),
    ("Euler", "euler"),
    ("Euler Ancestral", "euler_ancestral"),
    ("DDIM", "ddim"),
]


# ============================================================
#  CSS Styling
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2em;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 1.5em;
}
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.3s;
}
.image-compare {
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.control-image {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}
.error-box {
    border: 1px solid #f44336;
    background-color: #ffebee;
    padding: 12px;
    border-radius: 8px;
    color: #b71c1c;
}
.success-box {
    border: 1px solid #4caf50;
    background-color: #e8f5e9;
    padding: 12px;
    border-radius: 8px;
    color: #1b5e20;
}
.param-section {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}
"""


# ============================================================
#  Application State
# ============================================================

class AppState:
    """Manages the application's singleton pipeline instances."""

    _engine: Optional[ControlNetInference] = None
    _scenario_pipeline: Optional[ScenarioPipeline] = None
    _initialized: bool = False
    _init_error: Optional[str] = None

    @classmethod
    def initialize(cls, progress=gr.Progress()) -> bool:
        """Initialize GPU engine and scenario pipeline. Call once at startup."""
        if cls._initialized:
            return True

        try:
            if not torch.cuda.is_available():
                cls._init_error = "❌ CUDA GPU 不可用，请检查 PyTorch CUDA 安装。"
                return False

            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

            progress(0.1, desc="加载 Stable Diffusion 基础模型...")

            # Initialize engine
            cls._engine = ControlNetInference(
                sd_model_path=SD_MODEL_PATH,
                device="cuda",
                torch_dtype=torch.float16,
                local_files_only=True,
                enable_attention_slicing=True,
                enable_vae_slicing=True,
                disable_safety_checker=True,
            )

            progress(0.4, desc="初始化场景管线...")

            # Initialize scenario pipeline
            cls._scenario_pipeline = ScenarioPipeline(
                engine=cls._engine,
                sd_models_dir=MODELS_DIR,
            )

            cls._initialized = True
            cls._init_error = None

            progress(1.0, desc="初始化完成 ✓")
            logger.info("AppState initialized successfully")
            return True

        except Exception as e:
            cls._init_error = f"初始化失败: {str(e)}"
            logger.error(f"Initialization error: {traceback.format_exc()}")
            return False

    @classmethod
    def get_engine(cls) -> Optional[ControlNetInference]:
        return cls._engine

    @classmethod
    def get_scenario_pipeline(cls) -> Optional[ScenarioPipeline]:
        return cls._scenario_pipeline

    @classmethod
    def is_ready(cls) -> bool:
        return cls._initialized and cls._engine is not None


# ============================================================
#  Callback Functions
# ============================================================

def on_scenario_change(scenario_id: str):
    """Update style dropdown when scenario changes."""
    if scenario_id in STYLE_CHOICES:
        choices = STYLE_CHOICES[scenario_id]
        return gr.update(choices=choices, value=choices[0][1], visible=True)
    return gr.update(choices=[], value=None, visible=False)


def on_generate(
    scenario_id: str,
    style_id: str,
    input_image,
    prompt_text: str,
    negative_prompt_text: str,
    steps: int,
    cfg_scale: float,
    cn_scale_1: float,
    cn_scale_2: float,
    cn_scale_3: float,
    scheduler: str,
    seed: int,
    progress=gr.Progress(),
):
    """Main generation callback."""
    # Validate inputs
    if input_image is None:
        return None, None, None, None, "⚠️ 请先上传一张图片"

    if not AppState.is_ready():
        return None, None, None, None, f"❌ 系统未初始化: {AppState._init_error or '未知错误'}"

    try:
        sp = AppState.get_scenario_pipeline()
        config = sp.get_config(scenario_id)

        # Build list of ControlNet scales based on scenario
        n_controlnets = len(config.controlnet_names)
        all_scales = [cn_scale_1, cn_scale_2, cn_scale_3]
        cn_scales = all_scales[:n_controlnets]

        # Normalize scales to match what the pipeline expects
        if n_controlnets == 1:
            cn_scales = cn_scales[0]
        # For 2+ ControlNets, pass the list

        # Set seed (0 = random)
        actual_seed = None if seed == 0 else seed

        # Determine prompt
        prompt = prompt_text.strip() if prompt_text.strip() else None
        negative = negative_prompt_text.strip() if negative_prompt_text.strip() else None

        progress(0.1, desc="预处理中...")

        result = sp.run(
            scenario=scenario_id,
            image=input_image,
            style=style_id,
            prompt=prompt,
            negative_prompt=negative,
            steps=steps,
            cfg=cfg_scale,
            controlnet_scales=cn_scales if isinstance(cn_scales, list) else None,
            scheduler=scheduler,
            seed=actual_seed,
            width=512,
            height=512,
        )

        output_image = result["output_image"]
        control_images = result["control_images"]
        used_seed = result["seed"]
        elapsed = result["time"]

        # Prepare control image display
        control_gallery = []
        cn_names = config.controlnet_names
        for i, cimg in enumerate(control_images):
            label = cn_names[i] if i < len(cn_names) else f"control_{i}"
            control_gallery.append((cimg, f"{label}"))

        status_msg = (
            f"✅ 生成完成 | 种子: {used_seed} | "
            f"耗时: {elapsed:.1f}s | "
            f"场景: {config.name} | "
            f"风格: {style_id}"
        )

        return output_image, control_gallery, used_seed, elapsed, status_msg

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Generation error: {error_detail}")
        return (
            None, None, None, None,
            f"❌ 生成失败: {str(e)}\n\n详细信息已记录到日志，请检查参数后重试。"
        )


# ============================================================
#  UI Construction
# ============================================================

def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        title="SD ControlNet – 可控图像生成系统",
        fill_height=False,
    ) as app:

        # ---- Header ----
        gr.HTML("""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            <h1 class="title">🎨 Stable Diffusion + ControlNet</h1>
            <p class="subtitle">可控图像生成系统 – 线稿上色 | 草图转写实 | 真人转动漫 | 老照片修复</p>
        </div>
        """)

        # ---- Status Bar ----
        status_text = gr.Markdown(
            "🔄 正在初始化系统...",
            elem_classes=["param-section"],
        )

        # ---- Main Layout ----
        with gr.Row(equal_height=False):
            # ---- Left Column: Input ----
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 📥 输入")

                input_image = gr.Image(
                    label="上传图片",
                    type="pil",
                    height=350,
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    elem_classes=["upload-area"],
                )

                with gr.Row():
                    scenario_dropdown = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value="lineart_coloring",
                        label="🎯 应用场景",
                        interactive=True,
                    )

                style_dropdown = gr.Dropdown(
                    choices=STYLE_CHOICES["lineart_coloring"],
                    value="vivid_anime",
                    label="🎭 风格选择",
                    interactive=True,
                )

                with gr.Accordion("📝 提示词设置", open=False):
                    prompt_input = gr.Textbox(
                        label="正向提示词 (留空则使用默认模板)",
                        placeholder="自定义提示词...",
                        lines=3,
                    )
                    negative_prompt_input = gr.Textbox(
                        label="负向提示词 (留空则使用默认)",
                        placeholder="要避免的内容...",
                        lines=2,
                    )

            # ---- Right Column: Output and Parameters ----
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📤 生成结果")

                output_image = gr.Image(
                    label="生成结果",
                    type="pil",
                    height=350,
                    image_mode="RGB",
                    elem_classes=["image-compare"],
                )

                with gr.Accordion("🔧 高级参数", open=True):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=25,
                            step=1,
                            label="🔄 采样步数",
                            info="更多步数 = 更精细但更慢",
                        )
                        cfg_slider = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="🎯 CFG 引导强度",
                            info="越高越贴近提示词，但可能失真",
                        )

                    gr.Markdown("##### ControlNet 权重 (按场景使用)")

                    with gr.Row():
                        cn_scale_1 = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.85,
                            step=0.05,
                            label="ControlNet #1",
                            info="Lineart / Scribble / AnimeLineart / Canny",
                        )
                        cn_scale_2 = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.75,
                            step=0.05,
                            label="ControlNet #2",
                            info="Canny / OpenPose / Depth",
                        )
                        cn_scale_3 = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.05,
                            label="ControlNet #3",
                            info="Depth / 额外控制",
                        )

                    with gr.Row():
                        scheduler_dropdown = gr.Dropdown(
                            choices=SCHEDULER_CHOICES,
                            value="dpm++",
                            label="⚙️ 采样器",
                        )
                        seed_input = gr.Number(
                            label="🎲 随机种子 (0=随机)",
                            value=0,
                            precision=0,
                            minimum=0,
                        )

                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 开始生成",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    clear_btn = gr.Button(
                        "🗑️ 清空",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

        # ---- Control Images Gallery ----
        with gr.Accordion("🔍 预处理控制图 (ControlNet 输入)", open=False):
            control_gallery = gr.Gallery(
                label="ControlNet 条件图",
                columns=3,
                height=200,
                object_fit="contain",
                elem_classes=["control-image"],
            )

        # ---- Info Row ----
        with gr.Row():
            info_seed = gr.Number(label="实际种子", value=0, interactive=False, scale=1)
            info_time = gr.Number(label="耗时 (秒)", value=0, interactive=False, scale=1)

        # ---- Event Handlers ----

        scenario_dropdown.change(
            fn=on_scenario_change,
            inputs=[scenario_dropdown],
            outputs=[style_dropdown],
        )

        # Update default parameters when scenario changes
        def on_scenario_update_params(scenario_id):
            """Update sliders to scenario-appropriate defaults."""
            defaults = {
                "lineart_coloring": (25, 8.0, 0.9, 0.65, 0.7, "dpm++"),
                "sketch_to_realistic": (30, 9.0, 0.95, 0.75, 0.7, "dpm++"),
                "photo_to_anime": (30, 8.5, 0.8, 0.6, 0.7, "dpm++"),
                "old_photo_restore": (25, 7.5, 0.85, 0.75, 0.7, "ddim"),
            }
            params = defaults.get(scenario_id, (25, 7.5, 0.85, 0.75, 0.7, "dpm++"))
            return (
                gr.update(value=params[0]),   # steps
                gr.update(value=params[1]),   # CFG
                gr.update(value=params[2]),   # CN scale 1
                gr.update(value=params[3]),   # CN scale 2
                gr.update(value=params[4]),   # CN scale 3
                gr.update(value=params[5]),   # scheduler
            )

        scenario_dropdown.change(
            fn=on_scenario_update_params,
            inputs=[scenario_dropdown],
            outputs=[
                steps_slider, cfg_slider,
                cn_scale_1, cn_scale_2, cn_scale_3,
                scheduler_dropdown,
            ],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                scenario_dropdown,
                style_dropdown,
                input_image,
                prompt_input,
                negative_prompt_input,
                steps_slider,
                cfg_slider,
                cn_scale_1,
                cn_scale_2,
                cn_scale_3,
                scheduler_dropdown,
                seed_input,
            ],
            outputs=[
                output_image,
                control_gallery,
                info_seed,
                info_time,
                status_text,
            ],
        )

        clear_btn.click(
            fn=lambda: (None, None, 0, 0, "", "🔄 已清空，请上传新图片"),
            inputs=[],
            outputs=[
                output_image,
                control_gallery,
                info_seed,
                info_time,
                prompt_input,
                status_text,
            ],
        )

        # ---- Initialization on Load ----
        def init_on_load(progress=gr.Progress()):
            success = AppState.initialize(progress)
            if success:
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return f"✅ 系统就绪 | GPU: {gpu_name} ({vram:.1f} GB) | 模型: SD 1.5 + ControlNet"
            else:
                return f"❌ 初始化失败: {AppState._init_error}"

        app.load(
            fn=init_on_load,
            outputs=[status_text],
        )

    return app


def launch_ui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    **kwargs,
):
    """Launch the Gradio web interface."""
    app = create_ui()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        **kwargs,
    )
    return app
