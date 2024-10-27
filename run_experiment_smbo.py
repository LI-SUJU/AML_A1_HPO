import ConfigSpace
from ConfigSpace import Constant
import numpy as np
import typing
import random
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
from random_search import RandomSearch
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from configuration_preprocess import configuration_preprocess_before_model_training, configuration_preprocess_before_sampling
from dataset_handler import get_dataframes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    parser.add_argument('--dataset', type=str, default='lcdb')
    
    return parser.parse_args()


def train_smbo(args, max_anchor, total_budget):
    
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    # df = pd.read_csv(args.configurations_performance_file)
    df = get_dataframes()[args.dataset]
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    
    
   
    
    anchor_size =  Constant("anchor_size", max_anchor)
    
    config_space.add(anchor_size)
    
    capital_phi = []
            
            
    for _ in range(20):  # Sample 20 initial configurations
        config = config_space.sample_configuration()
        theta_val = dict(config)
        theta_val = configuration_preprocess_before_sampling(config_space, theta_val)
        error_rate = surrogate_model.predict(theta_val)
        print(error_rate)
        capital_phi.append((theta_val, error_rate))
    
    smbo = SequentialModelBasedOptimization (config_space=config_space, max_anchor=max_anchor)
    
    smbo.initialize(capital_phi)
    
    budget_left = total_budget

    spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external = [], [], [], []
    # spearman_corr_internal[0], pvalue_internal[0] = smbo.get_spearman_correlation(df)
    # spearman_corr_external[0], pvalue_external[0] = surrogate_model.get_spearman_correlation()

    while budget_left:
        smbo.fit_model()
        theta_new = smbo.select_configuration() 
        performance = surrogate_model.predict(theta_new)
        smbo.update_runs((theta_new , performance))
        # smbo.all_performances.append(performance)
        spearman_corr_internal.append(smbo.get_spearman_correlation(df)[0])
        pvalue_internal.append(smbo.get_spearman_correlation(df)[1])
        spearman_corr_external.append(surrogate_model.get_spearman_correlation()[0])
        pvalue_external.append(surrogate_model.get_spearman_correlation()[1])

        budget_left = budget_left-1
    
        
    return smbo.result_performance, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external
    

def plotting(total_budget, all_performances, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external):
    
    # plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), all_performances, color='red', label='Best found configuration')
    

    
    min_performance = min(all_performances)

    min_budget = all_performances.index(min_performance)

    # plt.scatter(min_budget, min_performance, color='red', zorder=5, label=' Best Performance')

    # Add a horizontal line at the minimum performance point
    plt.axhline(y=min_performance, color='black', linestyle='--', label=f'Best performance: {min_performance:.6f}')
    # Get the current y-ticks
    yticks = list(plt.yticks()[0])

    # Add the minimum performance to the y-ticks if it's not already there
    if min_performance not in yticks:
        yticks.append(min_performance)
        yticks = sorted(yticks)  # Sort the ticks to keep them in order

    # Set the updated y-ticks
    plt.yticks(yticks)
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    title = f'Performances of SMBO on {args.dataset.upper()}'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    savapath = f'./plots/smbo_performance_{args.dataset.upper()}.png'
    plt.savefig(savapath)

    # Plot the Spearman correlation and p-value
    # plt.figure(figsize=(12, 12))
    
    # Plot Spearman correlation for internal and external models
    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), spearman_corr_internal, color='blue', label='Internal Model Spearman Correlation')
    plt.plot(range(total_budget), spearman_corr_external, color='green', label='External Model Spearman Correlation')
    plt.xlabel('Iteration')
    plt.ylabel('Spearman Correlation')
    title = f'Spearman Correlation of Internal and External Models on {args.dataset.upper()}'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = f'./plots/spearman_correlation_{args.dataset.upper()}.png'
    plt.savefig(save_path)
    
    # Plot p-value for internal and external models
    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), pvalue_internal, color='blue', label='Internal Model p-value')
    plt.plot(range(total_budget), pvalue_external, color='green', label='External Model p-value')
    plt.xlabel('Iteration')
    plt.ylabel('p-value')
    title = f'p-value of Internal and External Models on {args.dataset.upper()}'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = f'./plots/pvalue_{args.dataset.upper()}.png'
    plt.savefig(save_path)


    
if __name__ == "__main__":
    args = parse_args()
    max_anchor = 16000
    total_budget = 200 # 200
    result_performances,  spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external= train_smbo(args, max_anchor=max_anchor, total_budget=total_budget)
    plotting(total_budget, result_performances, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external)
    