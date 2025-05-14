"""
Copied results from lazy prediction:

                               Adjusted R-Squared  R-Squared  RMSE  Time Taken
Model
ExtraTreesRegressor                          0.20       0.22  0.89        0.68
HistGradientBoostingRegressor                0.20       0.22  0.89        0.72
RandomForestRegressor                        0.19       0.21  0.89        1.12
LGBMRegressor                                0.19       0.21  0.89        0.14
GradientBoostingRegressor                    0.18       0.20  0.90        0.57
NuSVR                                        0.17       0.18  0.90        0.12
MLPRegressor                                 0.16       0.18  0.91        0.90
SVR                                          0.15       0.17  0.91        0.13
BayesianRidge                                0.13       0.15  0.92        0.02
RidgeCV                                      0.13       0.15  0.92        0.02
ElasticNetCV                                 0.13       0.15  0.92        0.15
LassoCV                                      0.13       0.15  0.92        0.14
Ridge                                        0.13       0.15  0.92        0.02
LinearRegression                             0.13       0.15  0.92        0.02
LassoLarsCV                                  0.13       0.15  0.92        0.04
LassoLarsIC                                  0.13       0.15  0.92        0.03
Lars                                         0.13       0.15  0.92        0.02
TransformedTargetRegressor                   0.13       0.15  0.92        0.02
LarsCV                                       0.13       0.15  0.92        0.04
OrthogonalMatchingPursuitCV                  0.13       0.14  0.93        0.04
SGDRegressor                                 0.12       0.14  0.93        0.02
PoissonRegressor                             0.11       0.13  0.93        0.03
HuberRegressor                               0.11       0.13  0.93        0.05
XGBRegressor                                 0.10       0.12  0.94        0.23
TweedieRegressor                             0.10       0.11  0.94        0.03
GammaRegressor                               0.09       0.11  0.94        1.52
LinearSVR                                    0.08       0.10  0.95        0.03
KNeighborsRegressor                          0.07       0.09  0.95        0.05
BaggingRegressor                             0.05       0.07  0.96        0.18
OrthogonalMatchingPursuit                    0.02       0.05  0.98        0.01
DummyRegressor                              -0.02      -0.00  1.00        0.02
LassoLars                                   -0.02      -0.00  1.00        0.03
Lasso                                       -0.02      -0.00  1.00        0.02
ElasticNet                                  -0.02      -0.00  1.00        0.03
AdaBoostRegressor                           -0.05      -0.02  1.01        0.19
QuantileRegressor                           -0.12      -0.10  1.05        0.16
DecisionTreeRegressor                       -0.39      -0.36  1.16        0.06
RANSACRegressor                             -0.70      -0.66  1.29        0.30
ExtraTreeRegressor                          -0.77      -0.73  1.32        0.03
GaussianProcessRegressor                    -1.07      -1.02  1.42        0.25
PassiveAggressiveRegressor                  -1.12      -1.08  1.44        0.02
KernelRidge                                 -4.41      -4.29  2.30        0.12
"""


from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
import joblib
import json
import os

# Define models
models = {
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
}

# Hyperparameters to be tested for each model
# param_grids = {
#     'ExtraTreesRegressor': {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['auto', 'sqrt', 'log2']
#     },
#     'HistGradientBoostingRegressor': {
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_iter': [100, 200, 300],
#         'max_depth': [None, 10, 20],
#         'min_samples_leaf': [10, 20, 30],
#         'l2_regularization': [0.0, 0.1, 1.0],
#         'max_bins': [255, 512]
#     },
#     'RandomForestRegressor': {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['auto', 'sqrt', 'log2']
#     },
# }

# param_grid to save compute
param_grids = {
    'ExtraTreesRegressor': {
        'n_estimators': [10, 50],  # Reduced number of trees
        'max_depth': [None, 10],  # Limited depth
        'min_samples_split': [2, 5],  # Fewer options for splits
        'min_samples_leaf': [1, 2],  # Fewer options for leaf size
        'max_features': ['sqrt']  # Single option to reduce computation
    },
    'HistGradientBoostingRegressor': {
        'learning_rate': [0.1],  # Single learning rate
        'max_iter': [100],  # Fixed number of iterations
        'max_depth': [None, 10],  # Limited depth
        'min_samples_leaf': [20],  # Single option for leaf size
        'l2_regularization': [0.0],  # No regularization
        'max_bins': [255]  # Fixed number of bins
    },
    'RandomForestRegressor': {
        'n_estimators': [10, 50],  # Reduced number of trees
        'max_depth': [None, 10],  # Limited depth
        'min_samples_split': [2, 5],  # Fewer options for splits
        'min_samples_leaf': [1, 2],  # Fewer options for leaf size
        'max_features': ['sqrt']  # Single option to reduce computation
    },
}


def hyperparameter_tuning_grid_search_cv(X_train, X_test, y_train, y_test):
    """
    Perform GridSearchCV to decide on regression model to implement and the parameters to test.
    At the end of this script, we will have the best parameters saved as a .pkl file in the models directory.
    """
    # Dictionary for storing results
    results = {}
    search_name = "GridSearchCV"
    for model_name, model in models.items():

        results[model_name] = {}

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            scoring="r2",
            cv=5,
        )

        print(f"Performing {search_name} for {model_name}")
        
        # Perform hyperparameter search
        search.fit(X_train, y_train)

        # Best score and hyperparameters found
        best_params = search.best_params_
        best_score = search.best_score_

        file_name_params = f"model_params/params_{model_name}.pk1"
        os.makedirs(os.path.dirname(file_name_params), exist_ok=True)
        # export tuned params
        joblib.dump(search.best_params_, file_name_params)

        # dedicated model training and evaluation at later stage
        # Test on test data
        y_pred = search.predict(X_test)
        test_r2_score = r2_score(y_true=y_test, y_pred=y_pred)

        # Store results
        results[model_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_r2_score': test_r2_score
        }

    # export search results
    # make sure folders exist
    file_name_results = f"model_params/{search_name}_results.json"
    os.makedirs(os.path.dirname(file_name_results), exist_ok=True)
    with open(file_name_results, "w") as f:
        json.dump(results, f)
    return results