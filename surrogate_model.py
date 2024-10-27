import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.stats import spearmanr
from configuration_preprocess import configuration_preprocess_before_model_training, configuration_preprocess_before_sampling


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def get_spearman_correlation(self):
        # get the spearman correlation of self.model
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        # Assuming self.df is the dataframe used for training
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # Split the data in the same way as during training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict the test set
        y_pred = self.model.predict(X_test)
        # print('y_pred: {}'.format(y_pred))
        # print('y_test: {}'.format(y_test))
        # Calculate Spearman correlation
        spearman_corr, pvalue = spearmanr(y_test, y_pred)
        print('Spearman correlation: {}, p-value: {}'.format(spearman_corr, pvalue))

        return spearman_corr, pvalue

        

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        preprocessor = configuration_preprocess_before_model_training(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])



        model.fit(X_train, y_train)
        self.model = model

        y_pred = model.predict(X_test)
        print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
        print(f'R^2 Score: {r2_score(y_test, y_pred)}')


    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        # Fill missing keys in theta_new with default values from config_space
        theta_new = {**{key: self.config_space.get_hyperparameter(key).default_value for key in self.config_space}, **theta_new}

        # Transform theta_new into a dataframe and predict score
        df = pd.DataFrame([theta_new])
        y_pred = self.model.predict(df)

        return y_pred[0]
        # raise NotImplementedError()
