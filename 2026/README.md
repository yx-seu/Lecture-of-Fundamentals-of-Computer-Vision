# Lecture of Fundamentals of Computer Vision

![License](https://img.shields.io/badge/License-Public-green)
![Status](https://img.shields.io/badge/Status-Active-green)

## 📖 Project Overview

This is a **public repository** for the Computer Vision course final project, containing 9 comprehensive sub-projects covering fundamental and advanced topics in computer vision.

![Submission Deadline](images.png?raw=true)

## 🎯 Project Goals

- Demonstrate comprehensive understanding of fundamental computer vision concepts
- Apply theoretical knowledge to practical problem-solving scenarios
- Develop hands-on experience with state-of-the-art CV algorithms and libraries
- Showcase research and engineering capabilities through well-documented projects

## 📋 Repository Structure

```
final_project/
├── project_01_wangzhe/                 # 训练ViT-like model做物体识别（使用Imagenet数据集）
├── project_02_tangzhichao/             # 训练ViT-like model做物体分割（使用Imagenet数据集）
├── project_03_wangjian/                # SAM 2 自动掩码生成与数据集自动标注
├── project_04_chenyuqian/              # 借助CLIP做CIFAR10 零样本（zero shot）的 object detection
├── project_05_zhangjiancheng/          # table Diffusion+ControlNet 的可控图像生成 / 编辑
├── project_06_jiahongyi/               # YOLOv11在集成电路制造中的应用与优化
├── project_07_zhangyizhou/             # 基于Lenet实现mnist手写数字识别
├── project_08_haojiaheng/              # 肺部CT图像识别CNN网络的fpga部署
├── project_09_xudehao/                 # 基于YOLOv8的实时视频目标检测
└── README.md
```

## 📑 Project Submission Requirements

Each of the 9 sub-projects must include the following components:

### 1. **README Documentation** (English)
   - **Title**: Clear task title
   - **Project Objective**: Clear description of the problem to be solved
   - **Solution Approach**: Detailed explanation of your methodology and algorithm selection
   - **Instruction**: Clear installation steps and running instructions
   - **Results & Analysis**: Progressive results with figures and charts demonstrating step-by-step progress
   - **Conclusion**: Summary of findings, insights, and potential future improvements

### 2. **Source Code**
   - Clean, well-commented, and well-organized code
   - Include a `requirements.txt` or similar dependency file
   - If third-party code is used, **clearly acknowledge and cite the original source**
   - <mark>Present a sample inference in main.py (e.g., a few images or samples).</mark>

### 3. **Datasets** (if applicable)
   - Provide a **link to the dataset** (e.g., from Kaggle, official repositories, or cloud storage)
   - Include **at least 2-3 testing examples** with expected outputs
   - Document any preprocessing steps applied to the data
   - <mark>Do not put all datasets into the repository.</mark>

### 4. **Demo** 
   - <mark>**Provide a simple inference example in a Jupyter notebook that demonstrates and elaborates on the core functions of your project.**</mark>

## 📝 Sub-Project Structure Template

Each sub-project should follow this structure:

```
project_XX/
├── README.md              # Complete documentation (English)
├── src/                   # Source code directory
│   ├── main.py            # Main implementation
│   ├── utils.py           # Utility functions
│   └── requirements.txt    # Python dependencies
├── data/                  # Data directory
│   ├── test_examples/     # Testing examples
│   └── dataset_info.txt   # Dataset link and 
├── results/               # Output results
│   ├── figures/           # Visualization
│   └── tables/            # Visualization
├── demos/                 # Inference Demo
└── references.md          # Citation and attribution (if applicable)
```

## ✅ Submission Score Checklist

- [ ] Project README: Clear structure, complete installation & running instructions, rich examples (8 pts)
- [ ] Code Quality: Well-organized, commented, modular, readable, efficiency-aware (12 pts)
- [ ] Reproducibility: requirements.txt, dependencies installable, pretrained model correctly downloaded/loaded (10 pts)
- [ ] Demo / Inference Script: Runs successfully, reasonable output, polished (20 pts, gate item)
- [ ] Results + Progressive Visualizations: Progressive visualizations + in-depth analysis (15 pts)
- [ ] Technical Quality & Depth: Method justification, implementation quality, understanding of key code, simple ablation/comparative experiments (20 pts)
- [ ] Conclusion & Future Work: Genuine insights, thoughtful future work (10 pts)
- [ ] Academic Integrity: Proper citation of third-party code (5 pts)

## 📊 Expected Deliverables Timeline

- All projects should be completed by the final submission deadline
- Regular progress commits are encouraged
- Final documentation must be in English

## 🤝 Collaboration

- Each team is responsible for their designated sub-project
- Cross-team communication and collaboration are encouraged within the course

## 📧 Contact & Support

For questions regarding project requirements or technical support, please contact the teacher and teaching assistant in the group chat.

---

