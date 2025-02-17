import torch
import math
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import glob


# class RecommendDataset(Dataset):
#     def __init__(self, csv_path, device):
#         self.csv_path = csv_path
#         self.device = device
#         self.df_length = self._get_csv_length()  # Get total number of rows

#     def _get_csv_length(self):
#         """Get the number of rows in the CSV without loading it entirely."""
#         with open(self.csv_path, "r") as f:
#             return sum(1 for _ in f) - 1  # Subtract 1 for the header

#     def __len__(self):
#         return self.df_length
    
#     def __getitem__(self, idx):
#         """Load a single row dynamically to avoid memory issues."""
#         sample = pd.read_csv(self.csv_path, skiprows=idx+1, nrows=1)  # Read only one row
#         sample = sample.to_dict(orient="records")[0]  # Convert to dictionary
#         sample = {key: torch.tensor(val, dtype=torch.float32).to(self.device) for key, val in sample.items()}
#         return sample



# class RecommendDataset(IterableDataset):
#     def __init__(self, csv_path, device):
#         self.csv_path = csv_path
#         self.device = device

#     def __iter__(self):
#         chunk_size = 10000  # Load in batches
#         for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
#             for _, row in chunk.iterrows():
#                 sample = {key: torch.tensor(val, dtype=torch.float32).to(self.device) for key, val in row.items()}
#                 yield sample

from torch.utils.data import IterableDataset
class RecommendDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder_path, device, train_val):
        self.files = sorted(glob.glob(f"{folder_path}/{train_val}_data_chunk_*.csv"))
        self.device = device

    def process_file(self, file_path="/home/ubuntu/yhu/DeepFM_with_PyTorch/Data/chunks"):
        for df_chunk in pd.read_csv(file_path, chunksize=10000):
            for _, row in df_chunk.iterrows():
                sample = {key: torch.tensor(val, dtype=torch.float32) for key, val in row.items()}
                yield sample  

    def __iter__(self):
        for file in self.files:
            yield from self.process_file(file)


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
        


def train(data_path, category_count, category_dim, device):
    train_dataset = RecommendDataset(data_path, device=device, train_val="train")
    test_dataset = RecommendDataset(data_path, device=device, train_val="val")
    
    train_dataloader = DataLoader(train_dataset, batch_size=8192, num_workers=24)  #, shuffle=True
    test_dataloader = DataLoader(test_dataset, batch_size=8192, num_workers=24) #, shuffle=True
    
    criterion = nn.BCEWithLogitsLoss()
    model = ClickClassifier(category_count, category_dim).to(device) # Initialize the model and move it to the GPU
    # model.load_state_dict(torch.load("/home/ubuntu/yhu/DeepFM_with_PyTorch/Models/model_epoch_12_iter_3000.pth", map_location=device))
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_trainable_params = count_parameters(model)
    # print(f'Total trainable parameters: {total_trainable_params}')


    num_epochs = 100
    train_loss_list = []
    val_loss_list = []

    iteration = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count_batch = 0
        for i, inputs in enumerate(train_dataloader):
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), inputs["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.detach()
            print(f"Iteration: {iteration}, Loss: {loss}")
            # Save the model after every 3000 iterations
            if (i + 1) % 3000 == 0:
                torch.save(model.state_dict(), f"{project_path}/Models/model_epoch_{epoch}_iter_{i+1}.pth")
                print(f"Model saved at epoch {epoch}, iteration {i+1}")
            iteration += 1
            count_batch += 1

        train_loss_list.append(running_loss.detach().cpu().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/count_batch:.6f}")

        model.eval()
        testing_loss = 0.0
        count_batch = 0
        with torch.no_grad():  # Ensure no gradient tracking
            for inputs in test_dataloader:
                if count_batch == 100:
                    break
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), inputs['label'])
                auc = metrics.roc_auc_score(inputs["label"].detach().cpu().numpy(), outputs.detach().cpu().numpy())
                testing_loss += loss.detach()
                print(f"Test Batch: {count_batch}, Loss: {loss}")
                count_batch += 1
                
            val_loss_list.append(testing_loss.detach().cpu().numpy())
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {testing_loss/count_batch:.6f}")
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
    data_path = "/home/ubuntu/yhu/DeepFM_with_PyTorch/Data/chunks"
    project_path = "/home/ubuntu/yhu/DeepFM_with_PyTorch"

    # Check if GPU is available
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = pd.read_csv(f'{project_path}/Data/train_data.csv')
    val_data = pd.read_csv(f'{project_path}/Data/val_data.csv')
    test_data = pd.read_csv(f'{project_path}/Data/test_data.csv')
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
        category_dim[col_name] = 48

    # train_data_path = f'{project_path}/Data/train_data.csv'
    # val_data_path = f'{project_path}/Data/val_data.csv'
    train_loss_list, val_loss_list = train(data_path, category_count, category_dim, device)
    plot_loss(train_loss_list, val_loss_list)

    


        




        