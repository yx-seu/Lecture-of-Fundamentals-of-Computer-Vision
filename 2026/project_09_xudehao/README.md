# YOLOv8 实时视频目标检测项目

这是一个围绕 **YOLOv8 实时视频目标检测** 搭建的小型项目，包含：

- 摄像头或视频文件实时检测
- 推理延迟、FPS、检测数量等 benchmark
- 可重复的自动测试
- 纯 HTML benchmark 可视化报告
- 可替换模型、阈值、输入源和输出视频路径

默认主题是“通用目标检测演示”。如果要收窄成更具体的方向，可以直接替换视频来源和类别过滤，例如交通场景中的车辆/行人检测、课堂/会议室人数统计、生产线物体检测等。

## 项目结构

```text
src/yolov8_realtime/
  benchmark.py       # benchmark 运行和报告保存
  config.py          # 配置对象
  detector.py        # YOLOv8 封装
  metrics.py         # FPS / 延迟统计
  video.py           # 视频读取、标注、展示、保存
  visualization.py   # benchmark HTML 报告生成
scripts/
  detect_video.py            # 实时检测入口
  run_benchmark.py           # benchmark 入口
  visualize_benchmark.py     # 生成 HTML 报告
tests/
  test_*.py                  # 不依赖 YOLO/OpenCV 的自动测试
```

## 安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[runtime]"
```

如果只想先跑测试，不需要安装 YOLOv8 和 OpenCV：

```powershell
python -m unittest discover -s tests
```

## 实时检测

使用默认 webcam：

```powershell
python scripts/detect_video.py --source 0 --model yolov8n.pt
```

检测视频文件并保存标注结果：

```powershell
python scripts/detect_video.py --source data/demo.mp4 --model yolov8n.pt --output outputs/demo_annotated.mp4 --no-show
```

常用参数：

- `--conf`：置信度阈值，默认 `0.25`
- `--iou`：NMS IoU 阈值，默认 `0.45`
- `--imgsz`：输入尺寸，默认 `640`
- `--device`：运行设备，例如 `cpu`、`0`
- `--classes`：只检测指定 COCO 类别 id，例如 `--classes 0 2 3 5 7`
- `--max-frames`：最多处理多少帧，适合快速演示或测试

## Benchmark

真实 YOLOv8 benchmark 会生成合成视频帧并统计模型推理性能：

```powershell
python scripts/run_benchmark.py --model yolov8n.pt --frames 120 --warmup 10 --output-dir outputs/benchmarks
```

没有安装运行时依赖时，也可以跑 pipeline smoke benchmark：

```powershell
python scripts/run_benchmark.py --fake --frames 50 --output-dir outputs/benchmarks
```

生成可视化 HTML：

```powershell
python scripts/visualize_benchmark.py outputs/benchmarks/benchmark_latest.json --output outputs/benchmarks/report.html
```

## 自动测试

项目内置测试覆盖指标统计、benchmark 报告和 HTML 可视化生成：

```powershell
python -m unittest discover -s tests
```

也可以在安装 `pytest` 后使用：

```powershell
pytest
```

## 推荐演示流程

1. 用 `scripts/detect_video.py` 对 webcam 或视频文件做实时检测。
2. 用 `scripts/run_benchmark.py` 生成 benchmark JSON/CSV。
3. 用 `scripts/visualize_benchmark.py` 生成 HTML 报告。
4. 展示 annotated video、FPS/延迟曲线和 summary 指标。
