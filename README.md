<div align="center">

# XM-ALIGN: Unified Cross-Modal Embedding Alignment for Face-Voice Association

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg?style=for-the-badge)]()

**[English](README.md) | [简体中文](README_ZH.md)**

</div>

---

## 📅 Roadmap

- [x] **MAV-Celeb Dataset Preparation**: Released data processing scripts and directory specifications.
- [ ] **Training & Evaluation Code**: The complete pipeline is scheduled for release by the **end of 2025**.
- [ ] **VoxCeleb Extension**: Plans to support VoxCeleb cross-modal matching tasks in the future.

---

## 📁 Data Preparation

This project primarily focuses on the Face-Voice Association task based on the MAV-Celeb dataset. Please follow the steps below to prepare the data.

### Step 1: Download Data and Lists

Please download the raw dataset files and split lists from the following links:

| Content | Description | Download Link |
| :--- | :--- | :--- |
| **Dataset** | Includes raw audio and image data for v1 & v3 | [Google Drive: MAV-Celeb v1 & v3 datasets](https://drive.google.com/drive/folders/1OJyjXJULErvrvzLQmpJn5v8rRo0n_fod) |
| **Data Lists** | Includes train/test split files (.txt) | [Google Drive: MAV-Celeb v1 & v3 data lists](https://drive.google.com/drive/folders/1MEHtEVh9lSa9hNZxjEfNJnE3qrpm_PKw) |

> **Note**: You need to download `mavceleb_v1_train.zip`, `mavceleb_v1_test.zip`, `mavceleb_v3_tran.zip`, `mavceleb_v3_test.zip`, and the corresponding list folders.

### Step 2: Organize Directory Structure

After extracting the files, please organize the data strictly according to the following directory structure:

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

## 📧 Feedback
If you have any questions or encounter any issues, please feel free to submit an issue.