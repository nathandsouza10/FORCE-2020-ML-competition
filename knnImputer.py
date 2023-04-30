from sklearn.neighbors import KNeighborsRegressor
import pandas as pd


class knnImputer:
    def __init__(self, dataframe, missing_cols):
        self.df = pd.read_csv(filename)
