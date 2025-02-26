import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import glob
from sklearn import metrics
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
            
class ClickClassifier_DLRM(nn.Module):
    def __init__(self, category_count, category_dim):
        super(ClickClassifier_DLRM, self).__init__()
        self.category_count = category_count
        self.category_dim = category_dim
 
        # Embedding 
        categorical_cols = [f"C{i}" for i in range(1, 27)] + ["I1","I10","I11","I12"]
        integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
        self.embeddings = nn.ModuleDict({
            col_name: nn.Embedding(
                num_embeddings=self.category_count[col_name], 
                embedding_dim=self.category_dim[col_name]
            ) for col_name in categorical_cols
        })


        # Layers for processing component1: reshape sparse features from 128 to 64
        self.norm_sparse = nn.ModuleList()
        self.drop_sparse = nn.ModuleList()
        self.linear_sparse = nn.ModuleList()
        self.relu_sparse = nn.ModuleList()

        for i in range(30):
            self.norm_sparse.append(nn.BatchNorm1d(num_features=48))
            self.drop_sparse.append(nn.Dropout(p=0.2))
            self.linear_sparse.append(nn.Linear(48, 64))
            self.relu_sparse.append(nn.ReLU())


        # A layer for processing component2: reshape dense features from 9 to 64
        self.norm_dense = nn.BatchNorm1d(num_features=9)
        self.drop_dense = nn.Dropout(p=0.2)
        self.linear_dense = nn.Linear(9, 64)
        self.relu_dense = nn.ReLU()


        # neural netwrok layers
        self.total_dim = 961   
        self.norm_0 = nn.BatchNorm1d(num_features=self.total_dim)
        self.drop_0 = nn.Dropout(p=0.3)

        self.linear_1 = nn.Linear(self.total_dim, 500)
        self.relu_1 = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(num_features=500)
        self.drop_1 = nn.Dropout(p=0.3)

        self.linear_2 = nn.Linear(500, 250)
        self.relu_2 = nn.ReLU()
        self.norm_2 = nn.BatchNorm1d(num_features=250)
        self.drop_2 = nn.Dropout(p=0.3)

        self.linear_3 = nn.Linear(250, 100)
        self.relu_3 = nn.ReLU()
        self.norm_3 = nn.BatchNorm1d(num_features=100)
        self.drop_3 = nn.Dropout(p=0.3)

        self.linear_4 = nn.Linear(100, 1)


    def forward(self, x):
        ### Component 01: Use Neural Network Layer to adjust the shape for each of 
        #   the sparse feature from (256, 128) ---> (256, 64)
        # Extract embeddings and concatenate
        embedded_features = [self.embeddings[col_name](x[col_name].long()) for col_name in self.embeddings] # Length: number of sparse features
        # print("len of embbed_features list", len(embedded_features))   # 30
        # print("len of embbed_features list index0: ", embedded_features[0].shape)   #shape: (256, 128)

        sparse_outputs = []
        for i in range(30):
            sparse = self.norm_sparse[i](embedded_features[i])
            sparse = self.drop_sparse[i](sparse)
            sparse = self.linear_sparse[i](sparse)
            sparse = self.relu_sparse[i](sparse)
            sparse_outputs.append(sparse)
        # print("sparse_outputs[0].shape: ", sparse_outputs[0].shape) #shape: (256, 64), list length -len(sparse_output) = 30 


        ### Component 02: Use Neural Network Layer to adjust the shape for 
        #   the dense features from (256, 9) ---> (256, 64)
        ## Handle Dense Features:
        integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
        dense_features = [x[col_name].unsqueeze(1) for col_name in integer_cols]
        dense_pre = torch.cat(dense_features, dim=1)  # Shape: (batch_size, num_features, embedding_dim)
        # print("shape of dense_pre", dense_pre.shape)  #---> (256, 9)
        dense = self.norm_dense(dense_pre)
        dense = self.drop_dense(dense)
        dense = self.linear_dense(dense)
        dense = self.relu_dense(dense)
        # print("shape of dense", dense.shape)  #---> (256, 64)


        ### Concatenate processed dense and sparse features togheter
        sparse_outputs.append(dense)
        comb_sparse_dense = torch.stack(sparse_outputs, dim=1)  # Shape: (batch_size, num_features, embedding_dim)
        # print("shape of comb_sparse_dense", comb_sparse_dense.shape) #--> (256, 31, 64)

        ## Get the transposed matrix of comb_sparse_dense ---> comb_sparse_dense_T
        comb_sparse_dense_T = comb_sparse_dense.transpose(1, 2)  # Shape: (256, 64, 31)
        # print("shape of comb_sparse_dense_T", comb_sparse_dense_T.shape)

        ## Get the dot product between comb_sparse_dense and comb_sparse_dense_T ---> dp_matrix
        dp_matrix = torch.bmm(comb_sparse_dense, comb_sparse_dense_T)  # Shape: (256, 31, 31)
        # print("shape of dp_matrix", dp_matrix.shape)

        ## flatten the tensor dp_matrix (shape (256, 31, 31)) into a 2D matrix with shape (256, 961)
        dp_flat_matrix = dp_matrix.flatten(start_dim=1)  # Shape: (256, 961)
        # print("shape of dp_flat_matrix", dp_flat_matrix.shape)

        x = self.norm_0(dp_flat_matrix)
        x = self.drop_0(x)
        
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.norm_1(x)
        x = self.drop_1(x)

        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.norm_2(x)
        x = self.drop_2(x)

        x = self.linear_3(x)
        x = self.relu_3(x)
        x = self.norm_3(x)
        x = self.drop_3(x)

        x = self.linear_4(x)
        # print("x.shape: ", x.shape)

        return x