from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os


def categorize_categorical_quantitive_cols(df):
    """categorize categorical and quantitive cols of DataFrame"""
    cat_cols = pd.DataFrame.select_dtypes(df, include=["object"]).columns
    quant_cols = df.select_dtypes(exclude=["object"]).columns

    return cat_cols, quant_cols


def standard_scale_feature_data(X_train: pd.DataFrame, X_test: pd.DataFrame, dump_scaler: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    As you may notice, the data varies widely in scale, so normalization is necessary.
    You can use existing functions to construct this script.
    As output, this script will create two new datasets (X_train_scaled, X_test_scaled) which you will also save in data/processed.
    """
    _, quant_cols = categorize_categorical_quantitive_cols(X_train)

    scaler = StandardScaler()
    # scale quant_cols
    X_train[quant_cols] = scaler.fit_transform(X_train[quant_cols])
    X_test[quant_cols] = scaler.transform(X_test[quant_cols])

    if dump_scaler:
        file_name = "./data/joblib_data/scaler.pk1"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        joblib.dump(scaler, file_name)
    
    # export scaled df's
    # make sure folders exist
    file_name_x_train = "data/processed_data/X_train_scaled.csv"
    file_name_x_test = "data/processed_data/X_test_scaled.csv"
    
    os.makedirs(os.path.dirname(file_name_x_train), exist_ok=True)
    os.makedirs(os.path.dirname(file_name_x_test), exist_ok=True)
    
    X_train.to_csv(file_name_x_train)
    X_test.to_csv(file_name_x_test)

    return X_train, X_test
