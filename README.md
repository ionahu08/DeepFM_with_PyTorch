# DeepFM_with_PyTorch

A PyTorch implementation of DeepFM for CTR prediction problem.

## Usage

1. Download Criteo's Kaggle display advertising challenge dataset from [here] [1]

2. Generate a preprocessed dataset.

        ./Code/Preprocessing_data.ipynb

3. Train a DNN model and a DeepFM model,  predict.

        ./DNN_training.py
        ./deepFM_training.py

## Output


## Reference

- https://github.com/nzc/dnn_ctr.

- https://github.com/PaddlePaddle/models/tree/develop/deep_fm.

- DeepFM: A Factorization-Machine based Neural Network for CTR         Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

[1]: https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?file=10082655
