from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_benchmark_html(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_benchmark_html(report), encoding="utf-8")
    return output_path


def render_benchmark_html(report: dict[str, Any]) -> str:
    latencies = [float(value) for value in report.get("latencies_ms", [])]
    detections = [int(value) for value in report.get("detections_per_frame", [])]
    latency_svg = _line_chart(latencies, width=900, height=260, stroke="#2563eb", label="Latency ms")
    detection_svg = _bar_chart(detections, width=900, height=180, fill="#16a34a", label="Detections / frame")
    cards = _metric_cards(report)
    title = html.escape(str(report.get("model", "YOLOv8 Benchmark")))
    source = html.escape(str(report.get("source", "unknown")))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLOv8 Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dde5;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 14px;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); font-size: 14px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{
      padding: 14px;
      min-height: 88px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    section {{
      padding: 18px;
      margin-top: 16px;
      overflow-x: auto;
    }}
    svg {{
      width: 100%;
      max-width: 900px;
      height: auto;
      display: block;
    }}
    .axis {{ stroke: #a6afbd; stroke-width: 1; }}
    .grid-line {{ stroke: #e4e8ef; stroke-width: 1; }}
    .caption {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 720px) {{
      header {{ display: block; }}
      h1 {{ font-size: 22px; }}
      .metric-value {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>YOLOv8 Benchmark Report</h1>
        <div class="subtle">Model: {title}</div>
      </div>
      <div class="subtle">Source: {source}</div>
    </header>

    <div class="grid">
      {cards}
    </div>

    <section>
      <h2>推理延迟曲线</h2>
      {latency_svg}
      <div class="caption">越低越好；包含每一帧的 detector.predict 时间。</div>
    </section>

    <section>
      <h2>每帧检测数量</h2>
      {detection_svg}
      <div class="caption">用于观察输入内容或阈值变化对检测数量的影响。</div>
    </section>
  </main>
</body>
</html>
"""


def _metric_cards(report: dict[str, Any]) -> str:
    specs = [
        ("Frames", "frames"),
        ("Wall FPS", "fps_wall"),
        ("Inference FPS", "fps_inference_mean"),
        ("Mean Latency", "latency_ms_mean", " ms"),
        ("P95 Latency", "latency_ms_p95", " ms"),
        ("Detections", "detections_total"),
    ]
    cards: list[str] = []
    for label, key, *suffix in specs:
        value = report.get(key, 0)
        unit = suffix[0] if suffix else ""
        cards.append(
            "<div class=\"card\">"
            f"<div class=\"metric-label\">{html.escape(label)}</div>"
            f"<div class=\"metric-value\">{html.escape(str(value))}{html.escape(unit)}</div>"
            "</div>"
        )
    return "\n      ".join(cards)


def _line_chart(values: list[float], width: int, height: int, stroke: str, label: str) -> str:
    pad_left = 48
    pad_right = 18
    pad_top = 18
    pad_bottom = 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    if not values:
        return _empty_chart(width, height, "No latency data")
    y_max = max(values) if max(values) > 0 else 1.0
    y_min = min(values)
    span = max(y_max - y_min, 1.0)
    if len(values) == 1:
        points = [(pad_left, pad_top + plot_h / 2)]
    else:
        points = [
            (
                pad_left + index * plot_w / (len(values) - 1),
                pad_top + plot_h - ((value - y_min) / span) * plot_h,
            )
            for index, value in enumerate(values)
        ]
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    grid = _grid(width, height, pad_left, pad_top, plot_w, plot_h)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">
  {grid}
  <polyline fill="none" stroke="{stroke}" stroke-width="3" points="{point_text}" />
  <text x="{pad_left}" y="{height - 8}" font-size="12" fill="#5b6472">frames: {len(values)}</text>
  <text x="{pad_left}" y="14" font-size="12" fill="#5b6472">max {y_max:.2f} ms</text>
</svg>"""


def _bar_chart(values: list[int], width: int, height: int, fill: str, label: str) -> str:
    pad_left = 48
    pad_right = 18
    pad_top = 18
    pad_bottom = 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    if not values:
        return _empty_chart(width, height, "No detection data")
    y_max = max(max(values), 1)
    bar_w = max(plot_w / len(values), 1)
    bars = []
    for index, value in enumerate(values):
        height_px = (value / y_max) * plot_h
        x = pad_left + index * bar_w
        y = pad_top + plot_h - height_px
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 1, 1):.2f}" height="{height_px:.2f}" fill="{fill}" />')
    grid = _grid(width, height, pad_left, pad_top, plot_w, plot_h)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">
  {grid}
  {"".join(bars)}
  <text x="{pad_left}" y="{height - 8}" font-size="12" fill="#5b6472">frames: {len(values)}</text>
  <text x="{pad_left}" y="14" font-size="12" fill="#5b6472">max {y_max}</text>
</svg>"""


def _grid(width: int, height: int, pad_left: int, pad_top: int, plot_w: int, plot_h: int) -> str:
    x0 = pad_left
    y0 = pad_top + plot_h
    lines = [
        f'<line class="axis" x1="{x0}" y1="{pad_top}" x2="{x0}" y2="{y0}" />',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{width - 18}" y2="{y0}" />',
    ]
    for step in range(1, 4):
        y = pad_top + plot_h * step / 4
        lines.append(f'<line class="grid-line" x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" />')
    return "\n  ".join(lines)


def _empty_chart(width: int, height: int, message: str) -> str:
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(message)}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" fill="#5b6472">{html.escape(message)}</text>
</svg>"""

