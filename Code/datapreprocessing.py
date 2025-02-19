import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


# Convert train.txt&test.txt to train.csv&test.csv
def txt2csv(project_path):
    # Check how many rows are within the train.txt file
    with open(f"{project_path}/Data/train.txt", "r") as f:
        row_count = sum(1 for _ in f)
    print(f"Total rows from train.txt: {row_count}")

    # Define column names
    label_col = ["label"]
    int_feature_cols = [f"I{i}" for i in range(1, 14)]
    cat_feature_cols = [f"C{i}" for i in range(1, 27)]
    train_columns = label_col + int_feature_cols + cat_feature_cols
    test_columns = int_feature_cols + cat_feature_cols  # No label in test set

    # Convert train.txt to train.csv
    df_train = pd.read_csv(
        f"{project_path}/Data/train.txt", delimiter="\t", names=train_columns, 
        header=None, dtype={col: "Int64" for col in int_feature_cols}
    )
    df_train.to_csv(f"{project_path}/Data/train.csv", index=False)

    # Convert test.txt to test.csv
    df_test = pd.read_csv(
        f"{project_path}/Data/test.txt", delimiter="\t", names=test_columns, 
        header=None, dtype={col: "Int64" for col in int_feature_cols}
    )
    df_test.to_csv(f"{project_path}/Data/test.csv", index=False)



#read train and test csv files --> return train, test, concat data in dataFrame
def read_csv(project_path):
    int_feature_cols = [f"I{i}" for i in range(1, 14)]
    cat_feature_cols = [f"C{i}" for i in range(1, 27)]
    df_train = pd.read_csv(f"{project_path}/Data/train.csv", dtype={col: "Int64" for col in int_feature_cols}) 
    df_test = pd.read_csv(f"{project_path}/Data/test.csv", dtype={col: "Int64" for col in int_feature_cols})  
    
    df_all = pd.concat([df_train, df_test],axis=0)
    df_all = df_all.reset_index(drop=True)
    
    return df_train, df_test, df_all
    


# Replace Outlier with whiskers for I1 - I13
# Function to cap outliers based on Tukey's fences
def cap_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = 0
        upper_whisker = int(Q3 + 1.5 * IQR)
        
        df[col] = df[col].clip(lower=lower_whisker, upper=upper_whisker)
    return df



# Treat the following Integer Columns as Categorical Features: I1 & I10, I11, I12
    # Binning the values
    # I1:   -1: missing
        #    0: 0 <=I1 <2
        #    1: 2 <=I1 <4
        #    2: 4 <= I1 
    # I10:  -1: missing
        #    0: 0
        #    1: 1
        #    2: 2
    # I11:  replace missing with median value first
        #    0: 0 <=I11 <2
        #    1: 2 <=I11 <4
        #    2: 4 <= I11 
    # I12:  -1: missing
        #    0: 0
        #    1: 1
        #    2: 2 
# Binning for each of them
def bin_values(df):
    df["I1"] = pd.cut(df["I1"], bins=[-2, -1, 2, 4, float('inf')], labels=[-1, 0, 1, 2]).astype(int)
    df["I10"] = df["I10"].map({0: 0, 1: 1, 2: 2}).fillna(-1)
    df["I11"] = pd.cut(df["I11"], bins=[-float('inf'), 2, 4, float('inf')], labels=[0, 1, 2]).astype(int)
    df["I12"] = df["I12"].map({0: 0, 1: 1, 2: 2}).fillna(-1)
    return df

def Int2Cat(df):
    # replace missing values with -1 for I1, I10, and I12
    missing_cols = ["I1", "I10", "I12"]
    df[missing_cols] = df[missing_cols].fillna(-1)  # Fill with -1 as placeholder
    
    # replace missing values with median for I11
    df["I11"] = pd.to_numeric(df["I11"], errors="coerce")  # Convert to numeric
    df["I11"] = df["I11"].fillna(df["I11"].median())  # Fill with median

    # Binning
    df = bin_values(df)
    return df





# Treat the following Integer Columns as Integer Features: I2 - I9, I13
        #  Replace missing with median
        #  Log transform 
# Function to apply log transformation
def log_transform(df, cols):
    for col in cols:
        df[col] = np.log1p(df[col].clip(lower=0))   # log1p to handle zero values safely
    return df
    
def int2int(df):
    missing_cols = ["I2","I3","I4","I5","I6","I7","I8","I9","I13"]
    df[missing_cols] = df[missing_cols].astype(float).fillna(df[missing_cols].median())

    log_columns = ["I2","I3","I4","I6","I7","I8","I13"]  # no "I5", "I9"
    df = log_transform(df, log_columns)
    return df
    



# For the following categorical columns, we need to handle rare subcategories
# -----> Only keep subcategories whose percentage is over 5%
def rare_cat(threshold, df):    
    # List of columns with rare cases
    columns_with_rare_cases = ["C1", "C5", "C8", "C14", "C19", "C22", "C23", "C25"]
    
    # Process each specified column
    for col in columns_with_rare_cases:
        value_counts = df[col].value_counts(normalize=True)  # Get percentage of each subcategory
        rare_categories = value_counts[value_counts < threshold].index  # Identify rare subcategories
    
        # Replace rare subcategories with "rare"
        df[col] = df[col].replace(rare_categories, "rare") 
    return df




def data_preprocessing01(project_path):
    # Convert train.txt&test.txt to train.csv&test.csv
    txt2csv(project_path)

    # Read train and test csv files --> return train, test, concat data in dataFrame
    df_train, df_test, df_all = read_csv(project_path)

    # Replace Outlier with whiskers for I1 - I13; Function to cap outliers based on Tukey's fences
    integer_columns = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
    df_all = cap_outliers(df_all, integer_columns)

    # Treat the following Integer Columns as Categorical Features: I1 & I10, I11, I12
    # Handling missing + binning 
    df_all = Int2Cat(df_all)
    
    # Treat the following Integer Columns as Integer Features: I2 - I9, I13
    #  Replace missing with median + Log transform 
    df_all = int2int(df_all)
    
    # For the following categorical columns, we need to handle rare subcategories
    # -----> Only keep subcategories whose percentage is over 5%
    df_all = rare_cat(0.05, df_all)
        
    
    # Replace missing values with substring "Unknown"
    # Define categorical column names
    cat_feature_cols = [f"C{i}" for i in range(1, 27)]
    # Replace missing values with "Unknown"
    df_all[cat_feature_cols] = df_all[cat_feature_cols].fillna("Unknown")
    
    
    
    
    
    # for all categorical columns, label_encoding:
    categorical_cols = [f"C{i}" for i in range(1, 27)] + ["I1","I10","I11","I12"]
    
    # Apply Label Encoding
    label_encoders = {}  # Dictionary to store encoders for future transformations
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col])  # Encode categorical values
        label_encoders[col] = le  # Store the encoder for potential inverse transformations
    
    

    
    # Determine the split index
    split_index = int(len(df_all[df_all['label'].notnull()]) * 0.9)
    print(f"split_index is {split_index}")
    
    # Train data (Features)
    x_train = df_all.loc[df_all['label'].notnull(), :][:split_index]
    y_train = x_train['label']
    print("x_train shape:", x_train.shape)
    
    # Valid data (Features)
    x_val = df_all.loc[df_all['label'].notnull(), :][split_index:]
    print("x_val shape:", x_val.shape)
    
    # Test data (Features)
    x_test = df_all.loc[df_all['label'].isnull(), :]
    print("x_test shape:", x_test.shape)


    
    x_train.to_csv(f"{project_path}/Data/train_data.csv")
    x_val.to_csv(f"{project_path}/Data/val_data.csv")
    x_test.to_csv(f"{project_path}/Data/test_data.csv")





if __name__ == "__main__":
    project_path = "/home/ubuntu/yhu/DeepFM_with_PyTorch"

    # import train.txt and test.txt ---> train_csv and test.csv ---> train_data.csv/val_data.csv/test_data.csv
    data_preprocessing01(project_path)



    




    

