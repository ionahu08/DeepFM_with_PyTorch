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
   - Evaluate performance using **log loss** and **AUC**.  
   
   ```bash
   python DeepFM_training_GPU.py  # Imports model01_deepFM.py and RecommendDatasetClass.py
   python deepFM_training.py      # Imports model02_DLRM.py and RecommendDatasetClass.py



## Output


## Reference

- https://github.com/nzc/dnn_ctr.

- https://github.com/PaddlePaddle/models/tree/develop/deep_fm.

- DeepFM: A Factorization-Machine based Neural Network for CTR         Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

[1]: https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?file=10082655
