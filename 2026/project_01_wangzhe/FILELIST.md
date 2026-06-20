# ViT 课程作业提交清单

## 推荐提交方式

把以下文件保持原有相对路径，放入同一个压缩包：

```text
ViT课程作业_王喆_王宇辉_万宇强.zip
```

解压后应直接看到 `README.md`、`src/`、`demos/` 等，不要再嵌套多层无意义目录。

## 必须提交

```text
README.md
references.md
FILELIST.md
src/
  main.py
  vit.py
  infer.py
  utils.py
  eval_report.py
  requirements.txt
demos/
  inference_demo.ipynb
data/
  dataset_info.txt
  test_examples/
    example_1_tench.JPEG
    example_2_springer.JPEG
    example_3_cassette.JPEG
results/
  figures/
    accuracy_progression.png
    vit_architecture.png
    confusion_matrix.png
    per_class_accuracy.png
  tables/
    evaluation_summary.json
    per_class_accuracy.csv
reports/
  基于Vision Transformer的图像分类网络设计、训练与优化.docx
  training_improvements_v3_836.md
  final_eval_836/
    confusion_matrix.png
    per_class_accuracy.png
    per_class_accuracy.csv
    evaluation_summary.json
```

最终模型权重（`outputs_v3/vit_imagenette10_best.pt`，约 20MB）已通过邮件提交，不在仓库中。

## 建议提交（用于展示工作量）

```text
tests/
  test_vit.py
  test_infer.py
  test_eval_report.py
  test_pretrained_vit.py
scripts/
  generate_final80_assets.py
notebooks/
  vit_experiment_process.ipynb
reports/
  training_improvements_v2_80.md
  model_size_comparison.md
```

这部分能够说明我们不仅训练了最终模型，还完成了测试、ImageNet-1K 服务器尝试和实验过程记录。

## 通常不要提交

```text
data/imagenette2-160/        # 完整数据集 (~1.4GB)
outputs/                     # 旧权重
outputs_cosine20/
outputs_aug_finetune/
outputs_v2/
outputs_v3/                  # 最终权重已邮件提交
reports/server_pretrained_vit/  # 服务器文件
distill.py                   # 未完成的蒸馏代码
tests/test_distill.py
versions/                    # 旧版代码归档
findings.md, progress.md, task_plan.md, work_summary.md  # 开发日志
.agents/, AGENTS.md, CLAUDE.md
__pycache__/
.git/
```

## 最终训练命令

```bash
python vit.py \
  --data-root data/imagenette2-160 \
  --epochs 200 \
  --batch-size 32 \
  --num-workers 2 \
  --lr 3e-4 \
  --min-lr 1e-6 \
  --warmup-epochs 5 \
  --weight-decay 0.05 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --drop-path 0.1 \
  --scheduler cosine \
  --rand-augment \
  --random-erasing 0.25 \
  --output-dir outputs_v3 \
  --metrics-csv outputs_v3/metrics.csv
```

## 推理命令

```bash
# 演示模式（3 张测试图片）
python src/main.py

# 单张图片推理
python src/infer.py --image path/to/image.jpg --checkpoint path/to/checkpoint.pt --topk 5
```
