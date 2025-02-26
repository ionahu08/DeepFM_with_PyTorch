
import math
import numpy as np
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
import os
import re
import json

from RecommendDatasetClass import RecommendDataset
from model02_DLRM import ClickClassifier_DLRM

import multiprocessing as mp
mp.set_start_method("forkserver", force=True)



        
def train(data_path, category_count, category_dim, device):
    train_dataset = RecommendDataset(data_path, device=device, train_val="train")

    train_iter = iter(train_dataset)
    print(next(train_iter))  # Should print a sample

    test_dataset = RecommendDataset(data_path, device=device, train_val="val")
    
    train_dataloader = DataLoader(train_dataset, batch_size=8192, num_workers=18)  #, shuffle=True
    test_dataloader = DataLoader(test_dataset, batch_size=8192, num_workers=12) #, shuffle=True
    
    criterion = nn.BCEWithLogitsLoss()
    model = ClickClassifier_DLRM(category_count, category_dim).to(device) 
    # Initialize the model and move it to the GPU
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
                torch.save(model.state_dict(), f"{project_path}/Models/Models_DLRM/model_epoch_{epoch}_iter_{i+1}.pth")
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





def extract_epoch_iter(model_filename):
    """Extracts epoch and iteration numbers from filenames like 'model_epoch_5_iter_69000.pth'."""
    match = re.search(r"epoch_(\d+)_iter_(\d+)", model_filename)
    if match:
        epoch, iteration = int(match.group(1)), int(match.group(2))
        return epoch, iteration
    return None, None

def evaluate_model(model_path, project_path, category_count, category_dim, device):
    """Loads a model, evaluates it on the validation set, and returns validation loss and AUC."""
    criterion = nn.BCEWithLogitsLoss()
    model = ClickClassifier_DLRM(category_count, category_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = RecommendDataset(data_path, device=device, train_val="val")
    test_iter = iter(test_dataset)
    print(next(test_iter))  # Should print a sample
    test_dataloader = DataLoader(test_dataset, batch_size=8192, num_workers=12)

    y_true, y_pred = [], []
    testing_loss = 0.0
    count_batch = 0
    with torch.no_grad():
        for inputs in test_dataloader:
            if count_batch == 100:
                break
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), inputs['label'])
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            y_pred.extend(probs)
            y_true.extend(inputs["label"].cpu().numpy())
            testing_loss += loss.item()
            count_batch += 1
            print("count_batch",count_batch)
    val_loss = testing_loss / count_batch
    auc_score = metrics.roc_auc_score(y_true, y_pred)
    return val_loss, auc_score

    
def find_best_model(models_dir, project_path, category_count, category_dim, device):
    """Evaluates all models in the directory and finds the best one based on validation loss."""
    model_losses = {}

    for model_file in sorted(os.listdir(models_dir)):  # Ensure order
        if model_file.endswith(".pth"):
            epoch, iteration = extract_epoch_iter(model_file)
            if epoch is None:
                continue  # Skip if filename doesn't match pattern
            
            model_path = os.path.join(models_dir, model_file)
            val_loss, auc = evaluate_model(model_path, project_path, category_count, category_dim, device)
            
            model_losses[(epoch, iteration)] = {"val_loss": val_loss, "auc": auc}
            print(f"Model: {model_file}, Epoch: {epoch}, Iteration: {iteration}, Val Loss: {val_loss:.6f}, AUC: {auc:.4f}")

        with open(f"{project_path}/JSON/model_losses.json", "w") as f:
            json.dump({"model_losses": {f"{epoch}_{iteration}": v for (epoch, iteration), v in model_losses.items()}}, f)

    # plot_losses({epoch: model_losses[(epoch, iter)] for epoch, iter in model_losses})

    best_model = min(model_losses, key=lambda x: model_losses[x]["val_loss"])
    best_epoch, best_iteration = best_model
    print(f"Best Model: Epoch {best_epoch}, Iteration {best_iteration} with Val Loss: {model_losses[best_model]['val_loss']:.6f}")
    return best_model


def plot_log_loss(project_path, model_losses):
    """
    Plots the validation loss in log scale over training progress.

    Args:
        model_losses (dict): Dictionary with keys as "epoch_iteration" (e.g., "5_1200")
                             and values containing "val_loss".
    """
    epochs, iterations, losses = [], [], []

    for key, value in model_losses.items():
        epoch, iteration = map(int, key.split("_"))  # Convert string keys back to int
        epochs.append(epoch)
        iterations.append(iteration)
        losses.append(value["val_loss"])

    # Sort by epoch and iteration to maintain training order
    sorted_indices = np.argsort(np.array(epochs) * 1e6 + np.array(iterations))
    epochs = np.array(epochs)[sorted_indices]
    iterations = np.array(iterations)[sorted_indices]
    losses = np.array(losses)[sorted_indices]

    # Plot log loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs + iterations / max(iterations), losses, marker="o", linestyle="-", label="Validation Loss")
    plt.xlabel("Epoch + Iteration Progress")
    plt.ylabel("Validation Loss (Log Scale)")
    plt.yscale("log")  # Log scale for better visualization
    plt.title("Validation Loss Over Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.savefig(f"{project_path}/Plots/validation_loss_plot.png")
    print("Plot saved as validation_loss_plot.png")



 

if __name__ == "__main__":
    data_path = "/home/ubuntu/yhu/DeepFM_with_PyTorch/Data/chunks"
    project_path = "/home/ubuntu/yhu/DeepFM_with_PyTorch"

    # Check if GPU is available
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Read in train_data.csv/test_data.csv/val_data.csv
    # train_data = pd.read_csv(f'{project_path}/Data/train_data.csv')
    # val_data = pd.read_csv(f'{project_path}/Data/val_data.csv')
    # test_data = pd.read_csv(f'{project_path}/Data/test_data.csv')
    # all_data = pd.concat([train_data, val_data, test_data], axis=0)
    # print("Train Size:", train_data.shape)
    # print("Val Size:", val_data.shape)
    # print("Test Size:", test_data.shape)
    # print("All Size:", all_data.shape)

    # # Obtain category_count and category_dim, which will be used for NN embedding
    # categorical_cols = [f"C{i}" for i in range(1, 27)] + ["I1","I10","I11","I12"]
    # integer_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
    # category_count = {}
    # category_dim = {}
    # for col_name in categorical_cols:
    #     category_count[col_name] = int(all_data[col_name].max()+1)
    #     category_dim[col_name] = 48

    # # For first run Save category_count and category_dim
    # with open(f"{project_path}/JSON/category_meta.json", "w") as f:
    #     json.dump({"category_count": category_count, "category_dim": category_dim}, f)

    # For future run, read them directly
    with open(f"{project_path}/JSON/category_meta.json", "r") as f:
        meta_data = json.load(f)
    
    category_count = meta_data["category_count"]
    category_dim = meta_data["category_dim"]


    # # Training Model
    train_loss_list, val_loss_list = train(data_path, category_count, category_dim, device)

    # Find the best model, Visualize loss change
    # models_dir = "/home/ubuntu/yhu/DeepFM_with_PyTorch_https/Models/Models_DLRM"
    # best_model_epoch, best_model_iteration = find_best_model(models_dir, project_path, category_count, category_dim, device)
    
    # with open(f"{project_path}/JSON/model_losses.json", "r") as f:
    #     data = json.load(f)

    # import matplotlib
    # matplotlib.use('Agg')  # For non-GUI backends, comment out for local plots

    # model_losses = data["model_losses"]
    # plot_log_loss(project_path, model_losses)

    
    ## Used the saved model to predict, then use AUC score to evaluate model
    # evaluate(project_path, category_count, category_dim, device)
    # val_loss, auc_score = evaluate_model(f"{models_dir}/model_epoch_3_iter_6000.pth", project_path, category_count, category_dim, device)
    # print("model_epoch_3_iter_6000.pth, val_loss: ",val_loss, "auc", auc_score)  

    
    


        




        