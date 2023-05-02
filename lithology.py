import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


class Lithology:
    def __init__(self, filename, target_var):
        self.RAND_STATE = 42
        self.df = pd.read_csv(filename, sep=';')
        self.y_train = self.df.pop(target_var)
        self.X_train = self.df
        obj_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        self.encoder = LabelEncoder()
        for col in obj_cols:
            self.X_train[col] = self.encoder.fit_transform(self.X_train[col])

    def knn_impute(self, n_neighbours):
        # currently KNNImputer takes too long to compute
        knn_imputer = KNNImputer(n_neighbors=n_neighbours)
        imputed_data = knn_imputer.fit_transform(self.X_train, self.y_train)
        imputed_df = pd.DataFrame(imputed_data, columns=self.X_train.columns)
        return imputed_df

    def ffill_impute(self):
        return self.X_train.ffill(axis=1)

    def simple_impute(self):
        simple_imputer = SimpleImputer(strategy='mean')
        imputed_data = simple_imputer.fit_transform(self.X_train, self.y_train)
        imputed_df = pd.DataFrame(imputed_data, columns=self.X_train.columns)
        return imputed_df

    def regression_impute(self):
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=self.RAND_STATE)
        df_imputed = pd.DataFrame(imputer.fit_transform(self.X_train), columns=self.X_train.columns)
        return df_imputed

    def explore_imputed_data(self):
        ffil_imputed_data = self.ffill_impute()
        simple_imputed_data = self.simple_impute()
        regression_imputed_data = self.regression_impute()
        pca = PCA(n_components=2)
        simple_reduced_data = pca.fit_transform(simple_imputed_data, self.y_train)
        ffill_reduced_data = pca.fit_transform(ffil_imputed_data, self.y_train)
        regression_reduced_data = pca.fit_transform(regression_imputed_data, self.y_train)
        # plot imputed data using dimension reduction
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15))
        ax1.scatter(simple_reduced_data[:, 0], simple_reduced_data[:, 1], c=self.y_train, s=2)
        ax1.set_title('Simple Imputer')
        ax2.scatter(ffill_reduced_data[:, 0], ffill_reduced_data[:, 1], c=self.y_train, s=2)
        ax2.set_title('ffill Imputer')
        ax3.scatter(regression_reduced_data[:, 0], regression_reduced_data[:, 1], c=self.y_train, s=2)
        ax3.set_title('Linear Regression Imputer')
        plt.show()


l = Lithology(filename='./train/train.csv', target_var='FORCE_2020_LITHOFACIES_LITHOLOGY')
l.explore_imputed_data()
