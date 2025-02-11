import torch
import math
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing



class RecommendDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        sample = {key:torch.tensor(val) for key, val in sample.items()}
        return sample
    


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
        self.total_dim2 = 1000+961   # for combining component#1 interaction + component2 General NN
        

        # A layer for dense feature reshape from 9 to sparse_dim, e.g.128
        # in order to concatenation between sparse and dense features
        self.norm_dense = nn.BatchNorm1d(num_features=9)
        self.drop_dense = nn.Dropout(p=0.2)
        self.linear_dense = nn.Linear(9, 128)
        self.relu_dense = nn.ReLU()


        # A layer for processing component2
        self.norm_comp = nn.BatchNorm1d(num_features=self.total_dim)
        self.drop_comp = nn.Dropout(p=0.2)
        self.linear_comp = nn.Linear(self.total_dim, 1000)
        self.relu_comp = nn.ReLU()


        # neural netwrok layers
        self.norm_0 = nn.BatchNorm1d(num_features=self.total_dim2)
        self.drop_0 = nn.Dropout(p=0.3)

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
        general_x = self.norm_comp(general_x)
        general_x = self.drop_comp(general_x)
        general_x = self.linear_comp(general_x)
        general_x = self.relu_comp(general_x)  # Shape: (256, 1000)
        # print("shape of general_x: ", general_x.shape)


        ### Concatenate along axis=1: Component01(256, 961) + Component02(256, 1000) 
        x = torch.cat([general_x, dp_flat_matrix], dim=1)  # Shape: (256, 1961)
        # print("shape of x: ", x.shape)


        x = self.norm_0(x)
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

        return x
        


def train(train_data, test_data, category_count, category_dim):
    train_dataset = RecommendDataset(train_data)
    test_dataset = RecommendDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    model = ClickClassifier(category_count, category_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    train_loss_list = []
    val_loss_list = []

    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, inputs in enumerate(train_dataloader):
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), inputs["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss
            print(f"Iteration: {iteration}, Loss: {loss}")
            # Save the model after every 3000 iterations
            if (i + 1) % 3000 == 0:
                torch.save(model.state_dict(), f"./Models/model_epoch_{epoch}_iter_{i+1}.pth")
                print(f"Model saved at epoch {epoch}, iteration {i+1}")
            iteration += 1

        train_loss_list.append(running_loss.detach().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.6f}")

        model.eval()
        testing_loss = 0.0
        for inputs in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), inputs['label'])
            auc = metrics.roc_auc_score(inputs["label"].detach().numpy(), outputs.detach().numpy())
            testing_loss += loss
        
        val_loss_list.append(testing_loss.detach().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {testing_loss:.6f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], AUC Score: {auc:.6f}")
        print("____________________________________")

    return train_loss_list, val_loss_list


def plot_loss(train_loss, val_loss_list):

    plt.plot(list(range(1, len(train_loss)+1)), train_loss, label='Training Loss', color='blue')
    plt.plot(list(range(1, len(train_loss)+1)), val_loss_list, label='Validation Loss', color='orange')

    # Adding labels, legend, and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()


    # Set integer ticks on the x-axis
    plt.xticks(list(range(1, len(train_loss)+1)))
    # show the plow
    plt.show()
 


if __name__ == "__main__":
    train_data = pd.read_csv('./Data/dac/train_data.csv')
    val_data = pd.read_csv('./Data/dac/val_data.csv')
    test_data = pd.read_csv('./Data/dac/test_data.csv')
    all_data = pd.concat([train_data, val_data, test_data], axis=0)
    print("Train Size:", train_data.shape)
    print("Val Size:", val_data.shape)
    print("Test Size:", test_data.shape)
    print("All Size:", all_data.shape)

    categorical_cols = [f"C{i}" for i in range(1, 27)] + ["I1","I10","I11","I12"]
    integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
    category_count = {}
    category_dim = {}
    for col_name in categorical_cols:
        category_count[col_name] = int(all_data[col_name].max()+1)
        category_dim[col_name] = 128

    train_loss_list, val_loss_list = train(train_data, val_data, category_count, category_dim)
    plot_loss(train_loss_list, val_loss_list)

    


        




        