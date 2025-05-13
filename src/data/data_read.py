import pandas as pd


def read_csv_as_df(path: str) -> pd.DataFrame:
    """
    Read in csv and return data as pandas Dataframe.
    """ 
    return pd.read_csv(path)
