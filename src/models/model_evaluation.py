"""
Finally, using the trained model, we will evaluate its performance and make predictions.
At the end of this script, we will have a new dataset in data containing the predictions,
along with a scores.json file in the metrics directory that will capture evaluation metrics of our model (e.g., MSE, R2).
"""
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import json
import os

def evaluate_regression_model(model_name, model, X_test, y_test):
    # Predict on test set
    y_pred = model.predict(X_test)

    # R-Squared
    r2 = r2_score(y_test, y_pred)
    print("R-Squared: ",r2)

    # Adjusted R-Squared
    n = len(y_test)  # Number of samples
    p = X_test.shape[1]  # Number of features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("Adjusted R-Squared: ",adjusted_r2)

    # MSE
    mean_squared_error_reg = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print("MSE: ",mean_squared_error_reg)

    # RMSE
    root_mean_squared_error_reg = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
    print("RMSE: ",root_mean_squared_error_reg)

    scores = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "mean_squared_error": mean_squared_error_reg,
        "root_mean_squared_error": root_mean_squared_error_reg
    }

    # export metrics
    # make sure folders exist
    file_name = f"metrics/{model_name}.json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        json.dump(scores, f)
    return scores