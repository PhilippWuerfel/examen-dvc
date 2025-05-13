
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets. Our target variable is silica_concentrate, located in the last column of the dataset.
    This script will produce 4 datasets (X_test, X_train, y_test, y_train) that you can store in data/processed.
    """
    y = df[target]
    # removed date column here as not required for this project
    X = df.drop(columns=[target, 'date'])
    # 42 the answer for everything.
    return train_test_split(X, y, test_size=0.2, random_state=42)