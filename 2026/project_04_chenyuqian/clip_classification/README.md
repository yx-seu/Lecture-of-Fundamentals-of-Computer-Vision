# CLIP Latent Space Semantic Alignment Analysis

使用预训练 CLIP 模型分析 **图像特征与文本语义在共享 latent space 中的对齐关系**。CIFAR-10 zero-shot 分类仍然保留，但它在本项目中主要作为观测工具：如果 `"a photo of a cat."` 的文本向量附近确实聚集猫图像，并且狗、鹿、马等相近视觉语义对象也更容易进入邻域或发生混淆，就能更直观地说明 CLIP 学到的是图文语义对齐，而不是传统固定类别分类头。

全项目遵守：**不训练、不 fine-tune、不使用 CIFAR-10 训练集**。所有结果只来自冻结 CLIP image encoder 与 text encoder 的前向推理。

## 核心 Pipeline

```
CIFAR-10 image → CLIP image encoder → image feature vector (L2 norm)
                                                                    → cosine similarity → argmax → zero-shot prediction
class prompt text → CLIP text encoder → text feature vector (L2 norm)
```

## CLIP 双塔结构

- **Image Encoder (ViT-B/32)**: 将 224×224 图像编码为 512 维特征向量
- **Text Encoder (Transformer)**: 将类别描述文本编码为 512 维特征向量
- **Shared Latent Space**: 两个 encoder 输出的向量在同一个语义空间中可直接比较
- **Cosine Similarity**: 向量归一化后通过点积计算相似度
- **Semantic Anchor**: 例如 `"a photo of a cat."` 不是分类器权重，而是 latent space 中的文本语义锚点
- **Zero-shot**: 无需任何训练样本，直接比较图像与文本的语义相似度进行分类或检索

## 运行方法

```bash
pip install -r requirements.txt
python main.py
```

运行 cat-centered 语义对齐深化实验：

```bash
python cat_alignment_analysis.py
```

首次运行会自动下载：
- CIFAR-10 数据集 (~170MB)
- CLIP ViT-B/32 模型 (~600MB)

## 实验设计

| 实验 | Prompt 策略 |
|------|------------|
| Single Prompt | `"a photo of a {}."` |
| Multi-template Ensemble | 8 个模板取平均 |
| Cat Semantic Neighborhood | 计算所有 CIFAR-10 测试图像到 `"a photo of a cat."` 的相似度 |
| Fine-grained Cat Prompts | `domestic cat`, `tabby cat`, `Persian cat`, `Siamese cat`, `kitten`, `wild cat` |
| Misleading Prompts | `dog-like cat`, `toy cat`, `cartoon cat`, `not a cat` 等 |
| Optional External Cats | `data/custom_cats/` 与 `data/imagenet_cats/` 中的 evaluation-only 图片 |

## 输出结果

- **控制台输出**: 整体准确率、各类别准确率、分类报告 (precision/recall/F1)
- **confusion_matrix.png**: 带标注的混淆矩阵
- **misclassified_examples/**: 错误分类样例图片
- **outputs/cat_alignment/summary.json**: cat 语义邻域、细粒度 prompt、误导 prompt、外部图片分析摘要
- **outputs/cat_alignment/cat_similarity_ranking.csv**: 所有 CIFAR-10 测试图像按 cat 文本相似度排序
- **outputs/cat_alignment/cat_nearest_non_cat.csv**: 最接近 cat 文本向量的非 cat 图像
- **outputs/cat_alignment/cat_nearest_non_cat_montage.png**: cat 语义邻域中的非 cat 图像拼图
- **outputs/cat_alignment/cat_dog_deer_horse_pca.png**: cat/dog/deer/horse 图像特征与文本锚点的 PCA 可视化
- **outputs/cat_alignment/fine_grained_cat_prompts_on_cifar_cat.csv**: CIFAR-10 cat 图像在细粒度猫 prompt 上的相似度分布
- **outputs/cat_alignment/misleading_prompts_on_cifar_cat.csv**: 误导 prompt 对 cat 图像相似度的影响

## 预期结果

- 随机分类准确率: ~10% (10 类均匀分布)
- Single Prompt: 通常在 80-85%
- Multi-template Ensemble: 通常在 85-90%
- Cat 语义邻域中应主要出现真实 cat 图像，同时 dog 等视觉语义相近对象会比 airplane、ship 等对象更容易接近 cat 文本锚点
- 否定句和复杂关系 prompt 可能不能按人类逻辑方式工作，例如 `not a cat` 仍可能保留强烈的 cat 语义

## 文件结构

```
├── main.py                    # CIFAR-10 zero-shot 分类入口
├── cat_alignment_analysis.py  # cat-centered latent-space 语义对齐分析
├── model.py                   # CLIPZeroShotClassifier 模型封装
├── data_loader.py             # CIFAR-10 数据加载 (仅返回 PIL image)
├── prompt_templates.py        # Prompt 模板、细粒度 cat prompt、误导 prompt
├── evaluate.py                # 评估指标
├── data/
│   ├── custom_cats/           # 任意猫图 evaluation-only 输入目录
│   └── imagenet_cats/         # ImageNet cat 图片 evaluation-only 输入目录
├── outputs/cat_alignment/     # 新增语义对齐分析输出
├── requirements.txt           # 依赖
└── README.md                  # 本文件
```
