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

    from src.data.data_read import read_csv_as_df
    from src.data.data_split import split_data
    from src.data.data_normalisation import standard_scale_feature_data
    from src.models.lazy_model_selection import select_lazy_regression_model
    from src.models.grid_search import hyperparameter_tuning_grid_search_cv
    from src.models.model_training import train_extra_trees_regressor_model
    from src.models.model_evaluation import evaluate_regression_model
    # data input
    FILE_PATH = "data/raw_data/raw.csv"

    print(f"Reading in file: {FILE_PATH}")
    df = read_csv_as_df(FILE_PATH)
    
    # data splitting
    print("Splitting data (feature, target, train_test_split)")
    X_train, X_test, y_train, y_test = split_data(df=df, target="silica_concentrate")

    # data normalisation
    print("Normalisation of data")
    X_train_scaled, X_test_scaled = standard_scale_feature_data(X_train, X_test, dump_scaler=True)

    # optional: lazy model selection, analysis which can be used to configure GridSearch
    print("Pre Model Analysis with LazyPredict")
    select_lazy_regression_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # grid search for best parameters
    print("Hyperparameter tuning with GridSearchCV")
    search_res  = hyperparameter_tuning_grid_search_cv(X_train_scaled, X_test_scaled, y_train, y_test)
    # model training
    print("Model training")
    model = train_extra_trees_regressor_model(
        X_train_scaled,
        y_train,
        params=search_res["ExtraTreesRegressor"]["best_params"],
    )
    # model evaluation
    print("Model evaluation")
    model_evaluation = evaluate_regression_model("ExtraTreesRegressor", model, X_test_scaled, y_test)
    print(model_evaluation)
    print("Muchas Gracias for example ml process cycle")