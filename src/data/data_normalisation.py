from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib


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
        joblib.dump(scaler, "data/joblib_data/scaler.pk1")

    return X_train, X_test
