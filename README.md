# DeepFM_with_PyTorch

A PyTorch implementation of DeepFM for CTR prediction problem.

## Usage

1. **Download Dataset**  
   - Get Criteo’s Kaggle display advertising challenge dataset from [this link][1].  

2. **Preprocess the Data**  
   - **Exploratory Data Analysis (EDA):**  
     Use `./Notebook/preprocessing_data.ipynb` for EDA.  
   - **Data Preprocessing:**  
     Run `./Code/datapreprocessing.py` to process the dataset:  
     - Convert `train.txt` and `test.txt` into `train.csv` and `test.csv`.  
     - Further split `train.csv` into `train_data.csv`, `val_data.csv`, and `test_data.csv`.

3. **Train and Evaluate Models**  
   - Train a **DNN model**, a **DeepFM model**, and a **DLRM model**.  
   - Make predictions and select the best-performing model.  
   - Evaluate performance using **Binary Cross Entropy Loss** and **AUC**.  
   
   ```bash
   python DeepFM_training_GPU.py  # Imports model01_deepFM.py and RecommendDatasetClass.py
   python deepFM_training.py      # Imports model02_DLRM.py and RecommendDatasetClass.py



## Output


## Reference

- https://github.com/nzc/dnn_ctr.

- https://github.com/PaddlePaddle/models/tree/develop/deep_fm.

- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

- Deep Learning Recommendation Model for Personalization and Recommendation Systems, Maxim Naumov et al.

[1]: https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?file=10082655





---
title: "CTR Prediction using DeepFM"
output: github_document
---

# Criteo Display Ads Challenge - CTR Prediction with DeepFM (PyTorch)

This project implements a **PyTorch-based DeepFM model** to solve the click-through rate (CTR) prediction problem on display advertising data provided by Criteo.

## 🔗 References
- **Kaggle Competition Page**: [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data)  
- **Dataset Source (Figshare)**: [Download from Figshare](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?file=10082655)

---

## 🎯 Project Goal
Develop and evaluate deep learning models (DeepFM, DLRM, and DNN) for predicting CTR using large-scale advertising logs.

---

## 📂 Dataset Description

- **train.csv**: 36M+ rows of 7-day Criteo logs. Subsampled and ordered chronologically.
- **test.csv**: 6M+ rows from the following day.

**Features:**
- `Label`: Target (1 = clicked, 0 = not clicked)
- `I1`–`I13`: Integer features (mostly counts)
- `C1`–`C26`: Categorical features (hashed to 32-bit integers)

---

## ⚙️ Data Preprocessing

- See: `./Notebook/preprocessing_data.ipynb` (EDA) and `./Code/datapreprocessing.py` (processing + feature engineering)

**Steps:**
1. Merge `train.txt` and `test.txt`
2. Handle Integer Features:
   - Replace outliers with whiskers
   - Impute missing values with median
   - Binning 4 skewed features
   - Apply log-transform
3. Handle Categorical Features:
   - Replace rare values (<5%) with "Rare"
   - Fill missing with "Unknown"
   - Label Encoding
4. Split & Save:
   - `train_data.csv` (80%) → 400 chunks
   - `val_data.csv` (20%) → 90 chunks
   - `test_data.csv` (unlabeled) → 60 chunks

---

## 🤖 Model Training & Evaluation

### DeepFM Model
- Script: `DeepFM_training_GPU.py`  
- Model: `model01_deepFM.py`  
- Dataset: `RecommendDatasetClass.py`  

**Highlights:**
- GPU-enabled training
- Custom PyTorch `IterableDataset` for chunked CSV streaming
- BCEWithLogitsLoss + AdamW optimizer
- Model checkpointing every 3000 iterations
- Evaluation using AUC & loss on validation set
- Best checkpoint selected via validation loss

**Results:**
- ✅ `model_epoch_3_iter_6000.pth`
- 📉 Validation Loss: 0.4676
- 📈 AUC Score: 0.7901

### DLRM Model
- Script: `DLRM_training_GPU.py`  
- Model: `model02_DLRM.py`

**Architecture Differences:**
- 30 sparse + 9 dense features → 64-dim embeddings
- Pairwise interaction matrix (31 x 31 → 961-dim)
- Deep MLP with ReLU, BatchNorm, Dropout

---

## 📁 Repository Structure


