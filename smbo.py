import ConfigSpace
import numpy as np
import typing

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from configuration_preprocess import configuration_preprocess_before_model_training, configuration_preprocess_before_sampling
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from sklearn.impute import SimpleImputer


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space, max_anchor):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space
        self.R = None
        self.theta_inc =None
        self.theta_inc_performance = None
        self.internal_surrogate_model = None
        self.result_performance = []
        # self.random_search = RandomSearch(self.config_space)
        self.max_anchor = max_anchor
        # self.all_performances = []

    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        
        self.R = capital_phi
        self.theta_inc, self.theta_inc_performance = min(capital_phi, key=lambda x: x[1])

    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        configurations = [config for config, _ in self.R]
        X = pd.DataFrame(configurations)
        X= X.iloc[::]
        # imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'
        # X = imputer.fit_transform(X)
        y = np.array([performance for _, performance in self.R])

        preprocessor = configuration_preprocess_before_model_training(configurations)
        kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e1), nu=2.5)
        internal_surrogate_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        self.internal_surrogate_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', internal_surrogate_model)
        ]).fit(X, y)

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        
        sampled_configs = [
            configuration_preprocess_before_sampling(self.config_space, self.config_space.sample_configuration().get_dictionary())
            for _ in range(20)
        ]
        
        theta = pd.DataFrame(sampled_configs)
        
        ei_values = self.expected_improvement(self.internal_surrogate_model, self.theta_inc_performance, theta)
        
        return theta.iloc[np.argmax(ei_values)]

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        z = (f_star - mu) / sigma
        ei = (f_star - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        
        self.R.append(run)
        config, performance = run
        
        if performance < self.theta_inc_performance:
            self.theta_inc_performance = performance
            self.theta_inc = config
        
        self.result_performance.append(self.theta_inc_performance)