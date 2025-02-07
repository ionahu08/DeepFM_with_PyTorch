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
        self.total_dim = sum(list(category_dim.values())) + len(integer_cols)


        # neural netwrok layers
        self.norm_0 = nn.BatchNorm1d(num_features=self.total_dim)
        self.drop_0 = nn.Dropout(p=0.3)

        self.linear_1 = nn.Linear(self.total_dim, 1000)
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
        feature_dict = {col_name: self.embeddings[col_name](x[col_name].long()) for col_name in self.embeddings}

        integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
        for col_name in integer_cols:
            feature_dict[col_name] = x[col_name].unsqueeze(1)

        x = torch.cat(list(feature_dict.values()), dim=1)

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

    


        




        