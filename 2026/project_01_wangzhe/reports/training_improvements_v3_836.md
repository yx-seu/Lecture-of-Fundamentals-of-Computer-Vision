# 训练方法改进报告

> 作者：wanyuqiang | 日期：2026-06-19 | 训练环境：RTX 4060 Laptop (8GB VRAM)

---

## 一、概述

在不增加模型参数、不换数据集的前提下，通过改进训练方法，将 ViT 在 Imagenette-10 上的验证准确率从 **72.4% → 83.6%**（两轮迭代，累计 +11.2 个百分点）。

---

## 二、迭代过程

### 第一轮：v1 → v2（72.4% → 80.9%）

| 改动 | 说明 | 参数 |
|------|------|------|
| **Warmup 预热** | 前 5 个 epoch 学习率从 ~0 线性上升到 3e-4，避免训练早期震荡 | `--warmup-epochs 5` |
| **MixUp 混合增强** | 每批图片两两按比例混合，强制模型不背答案 | `--mixup-alpha 0.2` |
| **DropPath 随机深度** | 训练时随机丢弃整层残差路径，防止过拟合 | `--drop-path 0.1` |
| **训练更久** | 20ep → 100ep，配合 cosine 学习率衰减 | `--epochs 100` |

### 第二轮：v2 → v3（80.9% → 83.6%）

| 改动 | 说明 | 参数 |
|------|------|------|
| **CutMix 切块混合** | 与 MixUp 互补：每批随机选 MixUp(透明叠加) 或 CutMix(切块替换) | `--cutmix-alpha 1.0` |
| **训练更久** | 100ep → 200ep | `--epochs 200` |

---

## 三、最终结果

### 三代模型对比

| 类别 | v1 原始 | v2 (100ep) | v3 (200ep+CutMix) | 累计提升 |
|------|---------|-----------|-------------------|---------|
| tench (鱼) | 89.9% | 91.5% | **94.3%** | +4.4% |
| English springer (狗) | 80.5% | 89.6% | **92.9%** | +12.4% |
| cassette player (磁带机) | 68.6% | 84.0% | **88.0%** | +19.4% |
| chain saw (链锯) | 56.2% | 61.4% | **63.0%** | +6.8% |
| church (教堂) | 70.4% | 79.5% | **86.3%** | +15.9% |
| French horn (圆号) | 72.6% | 83.0% | **86.8%** | +14.2% |
| garbage truck (垃圾车) | 72.8% | 81.2% | **83.5%** | +10.7% |
| gas pump (油泵) | 62.5% | 69.9% | **71.1%** | +8.6% |
| golf ball (高尔夫球) | 67.4% | 83.2% | **87.0%** | +19.6% |
| parachute (降落伞) | 83.8% | 86.2% | **83.6%** | -0.2% |
| **整体** | **72.4%** | **80.9%** | **83.6%** | **+11.2%** |

### 训练指标对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| 验证准确率 | 72.4% | 80.9% | 83.6% |
| 平均每类准确率 | 72.5% | 81.0% | 83.7% |
| 训练 epoch | 20+10 | 100 | 200 |
| 模型参数 | 524 万 | 499 万 | 499 万 |
| 训练耗时 | ~15 分钟 | ~50 分钟 | ~1.5 小时 |

---

## 四、代码改动范围

所有改动集中在 `vit.py` 一个文件：

- 新增 `DropPath` 类（随机深度正则化）
- 新增 `mixup_batch()` 函数（图片透明混合）
- 新增 `cutmix_batch()` 函数（图片切块替换）
- 修改 `create_scheduler()` 支持 warmup 预热
- 修改 `VisionTransformer` 和 `TransformerBlock` 支持 DropPath
- 修改 `train_one_epoch()` 支持 MixUp + CutMix 可组合
- 新增 CLI 参数：`--warmup-epochs`、`--mixup-alpha`、`--cutmix-alpha`、`--drop-path`

其余文件（`infer.py`、`eval_report.py`、`tests/`）未动，完全兼容。

---

## 五、队友如何复现

### 方式 A：直接用训练好的模型

需要分享的文件：

```
├── vit.py                                      ← 改进后的训练代码（唯一改动的 .py 文件）
├── outputs_v3/vit_imagenette10_best.pt          ← v3 最佳模型 (83.6%)
└── data/imagenette2-160/                        ← 数据集（如队友已有则跳过）
```

操作步骤：

```bash
# 1. 把 vit.py 覆盖到项目目录
# 2. 确保 outputs_v3/vit_imagenette10_best.pt 在正确位置

# 3. 验证模型能加载
python vit.py --forward-only

# 4. 单张图推理测试
python infer.py --image 你的图片.jpg --checkpoint outputs_v3/vit_imagenette10_best.pt --topk 5

# 5. 生成完整评估报告（需要数据集）
python eval_report.py --checkpoint outputs_v3/vit_imagenette10_best.pt --data-root data/imagenette2-160 --output-dir reports/final_eval_v3 --metrics-csv outputs_v3/metrics.csv --batch-size 64
```

### 方式 B：从头训练复现 v3 结果

```bash
# 1. 把改好的 vit.py 覆盖到项目目录

# 2. 下载数据集 + 训练（200 epoch，约 1.5 小时）
python vit.py --download-imagenette --output-dir outputs_v3 --metrics-csv outputs_v3/metrics.csv --rand-augment --random-erasing 0.25
```

默认参数已包含所有改进：
- `--epochs 200`
- `--scheduler cosine`
- `--warmup-epochs 5`
- `--mixup-alpha 0.2`
- `--cutmix-alpha 1.0`
- `--drop-path 0.1`
- `--label-smoothing 0.1`
- `--min-lr 1e-6`

如果只想复现 v2 (100ep 80.9%)，加 `--epochs 100 --cutmix-alpha 0`。

### 环境要求

```bash
# PyTorch + torchvision (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install matplotlib tqdm scikit-learn
```

---

## 六、如果还想继续提升

| 方向 | 预期提升 | 说明 |
|------|---------|------|
| 训练 300ep+ | +1~2% | 模型可能还没完全收敛 |
| 蒸馏 | +2~4% | 服务器跑一个大模型当老师，教小模型 |
| 加大 Embedding (256→384) | +2~4% | 需要更大显存（你们租的服务器） |
| 扩展到 ImageNet-100 | 新赛道 | 100 类，需要服务器 |
