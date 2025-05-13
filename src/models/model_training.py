"""
Using the parameters found through GridSearch,
we will train the model and save the trained model in the models directory.
"""
from sklearn.ensemble import ExtraTreesRegressor
import joblib


def train_extra_trees_regressor_model(X_train, y_train, params: dict):
    """Train model based on params."""
    MODEL_NAME = "ExtraTreesRegressor"
    reg = ExtraTreesRegressor(**params)
    # fit model based on training data
    reg.fit(X_train, y_train)
    # # dump model
    joblib.dump(reg, f"models/{MODEL_NAME}.pk1")
    return reg