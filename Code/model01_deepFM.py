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
            
class ClickClassifier(nn.Module):
    def __init__(self, category_count, category_dim):
        super(ClickClassifier, self).__init__()
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
        self.total_dim = sum(list(category_dim.values())) + len(integer_cols)    # for general component2 General NN
        print("total_dim: ", self.total_dim)
        

        # A layer for dense feature reshape from 9 to sparse_dim, e.g.128
        # in order to concatenation between sparse and dense features
        self.norm_dense = nn.BatchNorm1d(num_features=9)
        self.drop_dense = nn.Dropout(p=0.2)
        self.linear_dense = nn.Linear(9, 48)
        self.relu_dense = nn.ReLU()


        # A layer for processing component2
        self.linear_comp_1 = nn.Linear(self.total_dim, 1000)
        self.relu_comp_1 = nn.ReLU()
        self.norm_comp_1 = nn.BatchNorm1d(num_features=1000)
        self.drop_comp_1 = nn.Dropout(p=0.2)

        self.linear_comp_2 = nn.Linear(1000, 500)
        self.relu_comp_2 = nn.ReLU()
        self.norm_comp_2 = nn.BatchNorm1d(num_features=500)
        self.drop_comp_2 = nn.Dropout(p=0.2)

        self.linear_comp_3 = nn.Linear(500, 100)
        self.relu_comp_3 = nn.ReLU()
        self.norm_comp_3 = nn.BatchNorm1d(num_features=100)
        self.drop_comp_3 = nn.Dropout(p=0.2)        

        self.total_dim2 = 100+961   # for combining component#1 interaction + component2 General NN

        # neural netwrok layers
        # self.norm_0 = nn.BatchNorm1d(num_features=self.total_dim2)
        # self.drop_0 = nn.Dropout(p=0.3)

        self.linear_1 = nn.Linear(self.total_dim2, 1000)
        self.relu_1 = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(num_features=1000)
        self.drop_1 = nn.Dropout(p=0.3)

        self.linear_2 = nn.Linear(1000, 500)
        self.relu_2 = nn.ReLU()
        self.norm_2 = nn.BatchNorm1d(num_features=500)
        self.drop_2 = nn.Dropout(p=0.3)

        self.linear_3 = nn.Linear(500, 100)
        self.relu_3 = nn.ReLU()
        self.norm_3 = nn.BatchNorm1d(num_features=100)
        self.drop_3 = nn.Dropout(p=0.3)

        self.linear_4 = nn.Linear(100, 1)

    def forward(self, x):
        ### Component 01: Dot Product --> get interaction beween sparse and dense features
        ## Handle Sparse Features:
        # Extract embeddings and concatenate
        embedded_features = [self.embeddings[col_name](x[col_name].long()) for col_name in self.embeddings] # Length: number of sparse features
        # print("len of embbed_features list", len(embedded_features))   
        new_matrix_sparse = torch.stack(embedded_features, dim=1)  # Shape: (batch_size, num_features, embedding_dim)
        # print("shape of new_matrix_sparse", new_matrix_sparse.shape) #--> (256, 30, 128)


        ## Handle Dense Features:
        integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
        dense_features = [x[col_name].unsqueeze(1) for col_name in integer_cols]
        new_matrix_dense_pre01 = torch.cat(dense_features, dim=1)  # Shape: (batch_size, num_features, embedding_dim)
        # print("shape of new_matrix_dense_pre01", new_matrix_dense_pre01.shape)  #---> (256, 9)
        new_matrix_dense_pre02 = self.norm_dense(new_matrix_dense_pre01)
        new_matrix_dense_pre02 = self.drop_dense(new_matrix_dense_pre02)
        new_matrix_dense_pre02 = self.linear_dense(new_matrix_dense_pre02)
        new_matrix_dense_pre02 = self.relu_dense(new_matrix_dense_pre02)
        # print("shape of new_matrix_dense_pre02", new_matrix_dense_pre02.shape)  #---> (256, 128)
        new_matrix_dense = new_matrix_dense_pre02.unsqueeze(1) 
        # print("shape of new_matrix_dense", new_matrix_dense.shape)  #---> (256, 1, 128)


        ## concatenate new_matrix_sparse and new_matrix_dense ---> new_matrix
        new_matrix = torch.cat([new_matrix_sparse, new_matrix_dense], dim=1)  # Shape: (256, 31, 128)
        # print("shape of new_matrix", new_matrix.shape)

        ## Get the transposed matrix of new_matrix ---> new_matrix_T
        new_matrix_T = new_matrix.transpose(1, 2)  # Shape: (256, 128, 31)
        # print("shape of new_matrix_T", new_matrix_T.shape)

        ## Get the dot product between new_matrix and new_matrix_T ---> dp_matrix
        dp_matrix = torch.bmm(new_matrix, new_matrix_T)  # Shape: (256, 31, 31)
        # print("shape of dp_matrix", dp_matrix.shape)

        ## flatten the tensor dp_matrix (shape (256, 31, 31)) into a 2D matrix with shape (256, 961)
        dp_flat_matrix = dp_matrix.flatten(start_dim=1)  # Shape: (256, 961)
        # print("shape of dp_flat_matrix", dp_flat_matrix.shape)


        ### Component 02: Our Features using general Neural Network
        feature_dict = {col_name: self.embeddings[col_name](x[col_name].long()) for col_name in self.embeddings}

        integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
        for col_name in integer_cols:
            feature_dict[col_name] = x[col_name].unsqueeze(1)

        general_x = torch.cat(list(feature_dict.values()), dim=1)
        general_x = self.linear_comp_1(general_x)
        general_x = self.relu_comp_1(general_x)
        general_x = self.norm_comp_1(general_x)
        general_x = self.drop_comp_1(general_x)

        general_x = self.linear_comp_2(general_x)
        general_x = self.relu_comp_2(general_x)
        general_x = self.norm_comp_2(general_x)
        general_x = self.drop_comp_2(general_x)

        general_x = self.linear_comp_3(general_x)
        general_x = self.relu_comp_3(general_x)
        general_x = self.norm_comp_3(general_x)
        general_x = self.drop_comp_3(general_x)
        # print("shape of general_x: ", general_x.shape)


        ### Concatenate along axis=1: Component01(256, 961) + Component02(256, 1000) 
        x = torch.cat([general_x, dp_flat_matrix], dim=1)  # Shape: (256, 1961)
        # print("shape of x: ", x.shape)


        # x = self.norm_0(x)
        # x = self.drop_0(x)
        
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

        return x