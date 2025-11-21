<div align="center">

# XM-ALIGN: Unified Cross-Modal Embedding Alignment for Face-Voice Association

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg?style=for-the-badge)]()

**[English](README.md) | [简体中文](README_ZH.md)**

</div>

---

## 📅 项目进度

- [x] **MAV-Celeb 数据集准备**：已发布数据处理脚本与目录规范。
- [ ] **训练与评估代码**：计划于 **2025年底** 发布完整 pipeline。
- [ ] **VoxCeleb 扩展**：计划在未来支持 VoxCeleb 跨模态匹配任务。

---

## 📁 数据准备

本项目主要基于 MAV-Celeb 数据集进行人脸-语音关联任务。请按照以下步骤准备数据。

### Step 1: 下载数据与列表

请从以下链接下载原始数据集文件及划分列表：

| 内容 | 说明 | 下载链接 |
| :--- | :--- | :--- |
| **Dataset** | 包含 v1 & v3 的原始音频与图像数据 | [Google Drive: MAV-Celeb v1 & v3 datasets](https://drive.google.com/drive/folders/1OJyjXJULErvrvzLQmpJn5v8rRo0n_fod) |
| **Data Lists** | 包含训练集与测试集的划分文件 (.txt) | [Google Drive: MAV-Celeb v1 & v3 data lists](https://drive.google.com/drive/folders/1MEHtEVh9lSa9hNZxjEfNJnE3qrpm_PKw) |

> **注意**：你需要下载 `mavceleb_v1_train.zip`, `mavceleb_v1_test.zip`, `mavceleb_v3_tran.zip`, `mavceleb_v3_test.zip` 以及对应的列表文件夹。

### Step 2: 目录结构整理

解压上述文件后，请严格按照以下目录结构整理数据：

```bash
data
├── v1                      # MAV-Celeb v1 Dataset
│   ├── faces
│   │   ├── English         # test set
│   │   ├── Urdu            # test set
│   │   ├── id0001          # train set (id folders)
│   │   └── idxxxx          # ...
│   └── voices
│       ├── English         # test set
│       ├── Urdu            # test set
│       ├── id0001          # train set (id folders)
│       └── idxxxx          # ...
├── v1_lists                # v1 Split Lists
│   ├── English_test.txt
│   ├── English_train.txt
│   ├── Urdu_test.txt
│   └── Urdu_train.txt
├── v3                      # MAV-Celeb v3 Dataset
│   ├── English_test        # test set
│   │   ├── face
│   │   └── voice
│   ├── German_test         # test set
│   │   ├── face
│   │   └── voice
│   ├── faces               # train set
│   └── voices              # train set
└── v3_lists                # v3 Split Lists
    ├── English_test.txt
    ├── English_train.txt
    ├── German_test.txt
    └── German_train.txt
```

## 📧 联系与反馈
有任何疑问欢迎提交 issue。