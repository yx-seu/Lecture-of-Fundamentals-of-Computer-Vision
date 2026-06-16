# Submission Notes

本目录是按老师 Git 上传教程整理后的干净提交版本。

## 已保留

- Python 源码：`main.py`、`model.py`、`data_loader.py`、`evaluate.py`、`prompt_templates.py`、`cat_alignment_analysis.py`
- 依赖和说明：`requirements.txt`、`README.md`
- 实验报告：`docs/experiment-report.pdf`、`.tex`、`.md`
- 关键实验结果：`confusion_matrix.png`、`outputs/cat_alignment/` 下的小型图表和摘要
- 少量 evaluation-only 外部猫图：`data/custom_cats/`、`data/imagenet_cats/`
- 错分样例：`misclassified_examples/`

## 已排除

- `data/cifar-10-batches-py/`：CIFAR-10 原始数据集，代码首次运行会自动下载
- `__pycache__/`：Python 缓存
- `docs/*.aux`、`docs/*.log`、`docs/*.out`、`docs/*.toc`：LaTeX 编译中间文件
- 模型权重类文件：`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.safetensors`、`*.bin`

老师要求：不要将大规模数据集和模型权重文件放入 GitHub，模型权重通过邮件发送 `220256729@seu.edu.cn`。本项目使用预训练 CLIP 并在运行时自动下载模型，没有本地自训练权重需要提交。

## 上传位置提醒

请把本目录内容放到你在课程仓库中对应的个人/小组文件夹下，例如教程中的 `project_01_XX`，并确认工作分支是你所在小组的 `group-X` 分支，不要提交到 `main` 分支。
