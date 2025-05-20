# CTR Prediction with DeepFM (PyTorch)

This project implements a PyTorch-based DeepFM model to solve the click-through rate (CTR) prediction task using Criteo's large-scale advertising dataset.


## Project Goal

Develop and evaluate deep learning models (DeepFM, DLRM) to predict ad click-through rate based on user behavior and ad features.

## Dataset Description

- **train.csv**: 36M+ rows from 7 days of logs. Chronologically ordered and subsampled.
- **test.csv**: 6M+ rows from the next day.

**Fields:**

- `Label`: Target variable (1 = clicked, 0 = not clicked)
- `I1–I13`: 13 integer (count) features
- `C1–C26`: 26 hashed categorical features

## Data Preprocessing

Files:
- EDA: `./Notebook/preprocessing_data.ipynb`
- Processing: `./Code/datapreprocessing.py`

Steps:
1. Merge `train.txt` and `test.txt`
2. Integer Features:
   - Replace outliers with whiskers
   - Fill missing values with median
   - Binning for 4 features
   - Log-transform skewed features
3. Categorical Features:
   - Replace rare values (<5%) with "Rare"
   - Fill missing values with "Unknown"
   - Label encoding
4. Split and chunk data:
   - `train_data.csv` (80%) → 400 chunks
   - `val_data.csv` (20%) → 90 chunks
   - `test_data.csv` → 60 chunks

## Model Training and Evaluation

### DeepFM Model

- Script: `DeepFM_training_GPU.py`
- Model: `model01_deepFM.py`
- Dataset Class: `RecommendDatasetClass.py`

Training Setup:
- GPU-enabled (if available)
- Chunked loading via custom `IterableDataset`
- BCEWithLogitsLoss, AdamW optimizer
- Batch size: 8192, `num_workers=24`
- Save checkpoints every 3000 iterations
- Validation with AUC and loss tracking

Best Model:
- `model_epoch_3_iter_6000.pth`
- Validation Loss: 0.4676
- AUC: 0.7901

### DLRM Model

- Script: `DLRM_training_GPU.py`
- Model: `model02_DLRM.py`

Architecture:
- 30 sparse + 9 dense → 64-dim embeddings
- Pairwise dot product → 961-dim interaction vector
- Deep MLP with ReLU, BatchNorm, and Dropout



## Conclusion

Among the models tested (DLRM, and DeepFM), the **DeepFM model achieved the best performance**. This is likely due to its architecture, which effectively combines:

- An **interaction layer** that captures feature interactions (like DLRM),
- And a **parallel MLP** that processes the concatenated raw embeddings and dense features.

This dual-path design allows DeepFM to model both low- and high-order feature interactions, making it particularly effective for CTR prediction tasks.



## References

- Kaggle Competition: [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data)  
- Dataset Downloaded From: [Figshare](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?file=10082655)
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
- Deep Learning Recommendation Model for Personalization and Recommendation Systems, Maxim Naumov et al.

