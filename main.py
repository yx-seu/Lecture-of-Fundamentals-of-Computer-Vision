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
  python main.py                    # 默认启动 Web UI (http://127.0.0.1:7860)
  python main.py --port 8080        # 使用端口 8080
  python main.py --share            # 生成公网分享链接
  python main.py --host 0.0.0.0     # 允许局域网访问
        """,
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="服务器绑定地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="服务器端口 (默认: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="生成 Gradio 公网分享链接"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="启用调试模式 (详细日志)"
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


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    # Launch Gradio UI
    from ui.app import create_ui, launch_ui

    logger.info(f"启动 Web UI: http://{args.host}:{args.port}")
    if args.share:
        logger.info("正在生成公网分享链接...")

    launch_ui(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
