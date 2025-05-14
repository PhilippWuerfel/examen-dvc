"""
Main entry point for the application.
Scripts will be programmed in a way that working directory is the root of the project.
"""
if __name__ == "__main__":

    import os
    import sys
    import pandas as pd
    # Dynamically set the root directory of the project
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT_DIR)  # Change the working directory to the project root
    sys.path.insert(0, ROOT_DIR)  # Add the project root to PYTHONPATH
    print(f"Current working directory: {os.getcwd()}")

    from src.models.model_training import train_extra_trees_regressor_model
    from utils import read_pk1_obj

    # load processed data
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    # load params json based on previous hyperparameter tuning
    model_params = read_pk1_obj("model_params/params_ExtraTreesRegressor.pk1")

    print("Model training")
    model = train_extra_trees_regressor_model(
        X_train_scaled,
        y_train,
        params=model_params,
    )
