#!/usr/bin/env python
"""
SD + ControlNet 可控图像生成系统
=================================
基于 Stable Diffusion 1.5 与 ControlNet 的可控图像生成系统。

四大应用场景:
  1. 线稿自动上色     – Lineart + Canny
  2. 草图转写实效果图 – Scribble
  3. 真人/实景转动漫风 – Anime Lineart + OpenPose + Depth
  4. 老照片修复       – Canny + Depth

启动方式:
  python main.py                    # 启动 Gradio Web UI
  python main.py --share            # 启动并生成公网分享链接
  python main.py --port 8080        # 指定端口
"""

# ================================================================
# CRITICAL: Patch gradio_client BEFORE any Gradio imports
# Gradio 4.44.1 + gradio_client 1.3.0 has a bug where
# _json_schema_to_python_type() crashes on boolean JSON schemas.
# ================================================================
def _apply_gradio_client_patch():
    try:
        from gradio_client import utils as gcu
        _orig = gcu._json_schema_to_python_type
        def _patched(schema, defs):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            return _orig(schema, defs)
        _patched.__name__ = "_json_schema_to_python_type"  # avoid re-patching
        gcu._json_schema_to_python_type = _patched
    except Exception:
        pass

_apply_gradio_client_patch()
# ================================================================

import os
import sys
import argparse
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SD + ControlNet 可控图像生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                 # 启动 Web UI (http://127.0.0.1:7860)
  python main.py --cli --all                     # 命令行模式: 运行所有场景样例推理
  python main.py --cli --scenario lineart_coloring --input my_sketch.png
  python main.py --port 8080                     # 指定端口
  python main.py --share                         # 生成公网分享链接
        """,
    )
    # Web UI arguments
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Generate Gradio public share link"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    # CLI inference arguments
    parser.add_argument(
        "--cli", action="store_true",
        help="Run in CLI mode (sample inference without Web UI)"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=["lineart_coloring", "sketch_to_realistic", "photo_to_anime", "old_photo_restore"],
        help="Select scenario for CLI inference"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input image for CLI inference"
    )
    parser.add_argument(
        "--style", type=str, default=None,
        help="Style preset for the selected scenario"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 4 scenario sample inferences on built-in test images"
    )
    parser.add_argument(
        "--steps", type=int, default=25,
        help="Number of inference steps (default: 25)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def print_banner():
    banner = r"""
   _____ _____         _____                _ _______ _
  / ____|  __ \       / ____|              | |__   __| |
 | (___ | |  | |______| |     ___  _ __ ___ | |_ | |  | |_
  \___ \| |  | |______| |    / _ \| '_ ` _ \| __|| |  | __|
  ____) | |__| |      | |___| (_) | | | | | | |_ | |  | |_
 |_____/|_____/        \_____\___/|_| |_| |_|\__||_|   \__|

    可控图像生成系统 – Stable Diffusion + ControlNet
    """
    print(banner)
    print("  线稿上色 | 草图转写实 | 真人转动漫 | 老照片修复")
    print("=" * 58)


def run_cli_inference(args):
    """Run sample inference in CLI mode (no Web UI)."""
    import torch
    from PIL import Image
    from src.pipeline.inference import ControlNetInference
    from src.pipeline.scenarios import ScenarioPipeline

    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    SD_PATH = os.path.join(MODELS_DIR, "stable-diffusion-v1-5")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test_examples")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_banner()
    print("[CLI Mode] Initializing inference engine...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    engine = ControlNetInference(
        sd_model_path=SD_PATH, device="cuda",
        torch_dtype=torch.float16, local_files_only=True,
    )
    sp = ScenarioPipeline(engine=engine, sd_models_dir=MODELS_DIR)

    if args.all:
        # Run all 4 scenarios on built-in test images
        test_cases = [
            {
                "scenario": "lineart_coloring",
                "method": sp.run_lineart_coloring,
                "input": os.path.join(DATA_DIR, "test_lineart_input.png"),
                "output": os.path.join(OUTPUT_DIR, "cli_lineart_coloring.png"),
                "style": "vivid_anime",
            },
            {
                "scenario": "sketch_to_realistic",
                "method": sp.run_sketch_to_realistic,
                "input": os.path.join(DATA_DIR, "test_sketch_input.png"),
                "output": os.path.join(OUTPUT_DIR, "cli_sketch_to_realistic.png"),
                "style": "architecture",
            },
            {
                "scenario": "photo_to_anime",
                "method": sp.run_photo_to_anime,
                "input": os.path.join(DATA_DIR, "test_generic_input.png"),
                "output": os.path.join(OUTPUT_DIR, "cli_photo_to_anime.png"),
                "style": "anime_film",
            },
            {
                "scenario": "old_photo_restore",
                "method": sp.run_old_photo_restore,
                "input": os.path.join(DATA_DIR, "test_oldphoto_input.png"),
                "output": os.path.join(OUTPUT_DIR, "cli_old_photo_restore.png"),
                "style": "restore_bw",
            },
        ]

        for i, tc in enumerate(test_cases):
            print(f"[{i+1}/4] Running: {tc['scenario']} (style: {tc['style']})")
            img = Image.open(tc["input"]).convert("RGB")
            result = tc["method"](image=img, style=tc["style"], steps=args.steps, seed=args.seed)
            result["output_image"].save(tc["output"])
            print(f"       Seed: {result['seed']} | Time: {result['time']:.1f}s")
            print(f"       Saved: {tc['output']}")
            print(f"       Prompt: {result.get('prompt_used', '')[:120]}...")
            print()

        print("=" * 60)
        print("All 4 sample inferences complete!")
        print(f"Outputs saved to: {OUTPUT_DIR}/")
        print("=" * 60)

    elif args.scenario:
        # Run a single specified scenario
        if not args.input:
            print("ERROR: --input <image_path> is required with --scenario")
            print("Usage: python main.py --cli --scenario lineart_coloring --input my_sketch.png")
            sys.exit(1)

        scenario_map = {
            "lineart_coloring": (sp.run_lineart_coloring, "vivid_anime"),
            "sketch_to_realistic": (sp.run_sketch_to_realistic, "architecture"),
            "photo_to_anime": (sp.run_photo_to_anime, "anime_film"),
            "old_photo_restore": (sp.run_old_photo_restore, "restore_bw"),
        }

        if not os.path.exists(args.input):
            print(f"ERROR: Input file not found: {args.input}")
            sys.exit(1)

        method, default_style = scenario_map[args.scenario]
        style = args.style or default_style

        print(f"Scenario: {args.scenario}")
        print(f"Style: {style}")
        print(f"Input: {args.input}")
        print(f"Steps: {args.steps} | Seed: {args.seed}")
        print()

        img = Image.open(args.input).convert("RGB")
        result = method(image=img, style=style, steps=args.steps, seed=args.seed)

        out_path = os.path.join(OUTPUT_DIR, f"cli_{args.scenario}.png")
        result["output_image"].save(out_path)
        print(f"Seed: {result['seed']} | Time: {result['time']:.1f}s")
        print(f"Saved: {out_path}")
        print()

    else:
        print("ERROR: --cli requires --all or --scenario <name>")
        print("Examples:")
        print("  python main.py --cli --all")
        print("  python main.py --cli --scenario lineart_coloring --input test.png")
        sys.exit(1)


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.cli:
        run_cli_inference(args)
        return

    print_banner()

    # Launch Gradio UI
    from src.ui.app import create_ui, launch_ui

    logger.info(f"Starting Web UI: http://{args.host}:{args.port}")
    if args.share:
        logger.info("Generating public share link...")

    launch_ui(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
