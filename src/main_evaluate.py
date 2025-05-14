"""
Main entry point for the application.
Scripts will be programmed in a way that working directory is the root of the project.
"""
if __name__ == "__main__":

    import os
    import sys
    # Dynamically set the root directory of the project
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT_DIR)  # Change the working directory to the project root
    sys.path.insert(0, ROOT_DIR)  # Add the project root to PYTHONPATH
    print(f"Current working directory: {os.getcwd()}")

    from src.models.model_evaluation import evaluate_regression_model
    from utils import read_pk1_obj
    import pandas as pd

    # load processed data
    X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")
    # load params json based on previous hyperparameter tuning
    model = read_pk1_obj("models/ExtraTreesRegressor.pk1")

    # model evaluation
    print("Model evaluation")
    model_evaluation = evaluate_regression_model("ExtraTreesRegressor", model, X_test_scaled, y_test)
    print(model_evaluation)
    print("Muchas Gracias for example ml process cycle")