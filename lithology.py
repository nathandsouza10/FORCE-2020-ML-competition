import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class Lithology:
    def __init__(self, filename, target_var):
        self.df = pd.read_csv(filename, sep=';')
        self.y_train = self.df.pop(target_var)
        self.X_train = self.df
        obj_cols = self.X_train.select_dtypes(include='object').columns.tolist()
        self.encoder = LabelEncoder()
        for col in obj_cols:
            self.X_train[col] = self.encoder.fit_transform(self.X_train[col])

    def knn_impute(self, n_neighbours):
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
        pass

    def explore_imputed_data(self):
        ffil_imputed_data = self.ffill_impute()
        simple_imputed_data = self.simple_impute()
        knn_imputed_data =self.knn_impute(n_neighbours=12)
        pca = PCA(n_components=2)
        simple_reduced_data = pca.fit_transform(simple_imputed_data, self.y_train)
        ffill_reduced_data = pca.fit_transform(ffil_imputed_data, self.y_train)
        knn_reduced_data = pca.fit_transform(knn_imputed_data, self.y_train)
        # plot imputed data using dimension reduction
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        ax1.scatter(simple_reduced_data[:, 0], simple_reduced_data[:, 1], c=self.y_train)
        ax1.set_title('Simple Imputer')
        ax2.scatter(ffill_reduced_data[:, 0], ffill_reduced_data[:, 1], c=self.y_train)
        ax2.set_title('ffill Imputer')
        ax3.scatter(knn_reduced_data[:, 0], knn_reduced_data[:, 1], c=self.y_train)
        ax3.set_title('KNN Imputer')
        plt.show()


l = Lithology(filename='./train/train.csv', target_var='FORCE_2020_LITHOFACIES_LITHOLOGY')
l.explore_imputed_data()
