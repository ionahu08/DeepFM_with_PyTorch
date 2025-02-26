import torch
import glob
import pandas as pd
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