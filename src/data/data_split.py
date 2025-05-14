
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(df: pd.DataFrame, target: str, dump_splits: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets. Our target variable is silica_concentrate, located in the last column of the dataset.
    This script will produce 4 datasets (X_train, X_test, y_train, y_test) that you can store in data/processed_data.
    """
    y = df[target]
    # removed date column here as not required for this project
    X = df.drop(columns=[target, 'date'])
    # 42 the answer for everything.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if dump_splits:
        # export df's
        # make sure folders exist
        file_name_x_train = "data/processed_data/X_train.csv"
        file_name_x_test = "data/processed_data/X_test.csv"
        file_name_y_train = "data/processed_data/y_train.csv"
        file_name_y_test = "data/processed_data/y_test.csv"
        
        os.makedirs(os.path.dirname(file_name_x_train), exist_ok=True)
        os.makedirs(os.path.dirname(file_name_x_test), exist_ok=True)
        os.makedirs(os.path.dirname(file_name_y_train), exist_ok=True)
        os.makedirs(os.path.dirname(file_name_y_test), exist_ok=True)

        X_train.to_csv(file_name_x_train)
        X_test.to_csv(file_name_x_test)
        y_train.to_csv(file_name_y_train)
        y_test.to_csv(file_name_y_test)

    return X_train, X_test, y_train, y_test