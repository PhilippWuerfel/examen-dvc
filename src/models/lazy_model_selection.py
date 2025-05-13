from lazypredict.Supervised import LazyRegressor
import pandas as pd

def select_lazy_regression_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    # reg = LazyRegressor(verbose=0, ignore_warnings=True)
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, regressors="all", predictions=False)
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    print(models.sort_values(by=['R-Squared'], ascending=False))