import ConfigSpace
from ConfigSpace import Constant
import numpy as np
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern
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
    df = get_dataframes()[args.dataset]
    
    surrogate = SurrogateModel(config_space)
    surrogate.fit(df)
    
    config_space.add(Constant("anchor_size", max_anchor))
    
    initial_configs = [
        (configuration_preprocess_before_sampling(config_space, dict(config_space.sample_configuration())), 
         surrogate.predict(configuration_preprocess_before_sampling(config_space, dict(config_space.sample_configuration()))))
        for _ in range(20)
    ]
    
    smbo = SequentialModelBasedOptimization(config_space=config_space, max_anchor=max_anchor)
    smbo.initialize(initial_configs)
    
    spearman_internal, pval_internal, spearman_external, pval_external = [], [], [], []
    
    for _ in range(total_budget):
        smbo.fit_model()
        new_config = smbo.select_configuration()
        performance = surrogate.predict(new_config)
        smbo.update_runs((new_config, performance))
        
        spearman_internal.append(smbo.get_spearman_correlation(df)[0])
        pval_internal.append(smbo.get_spearman_correlation(df)[1])
        spearman_external.append(surrogate.get_spearman_correlation()[0])
        pval_external.append(surrogate.get_spearman_correlation()[1])
    
    return smbo.result_performance, spearman_internal, pval_internal, spearman_external, pval_external
    

def plotting(total_budget, all_performances, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external):
    
    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), all_performances, color='red', label='Best found configuration')
    min_performance = min(all_performances)
    min_budget = all_performances.index(min_performance)
    plt.axhline(y=min_performance, color='black', linestyle='--', label=f'Best performance: {min_performance:.6f}')
    yticks = list(plt.yticks()[0])
    if min_performance not in yticks:
        yticks.append(min_performance)
        yticks = sorted(yticks)
    plt.yticks(yticks)
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title(f'Performances of SMBO on {args.dataset.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/smbo_performance_{args.dataset.upper()}.png')

    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), spearman_corr_internal, color='blue', label='Internal Model Spearman Correlation')
    plt.plot(range(total_budget), spearman_corr_external, color='green', label='External Model Spearman Correlation')
    plt.xlabel('Iteration')
    plt.ylabel('Spearman Correlation')
    plt.title(f'Spearman Correlation of Internal and External Models on {args.dataset.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/spearman_correlation_{args.dataset.upper()}.png')

    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), pvalue_internal, color='blue', label='Internal Model p-value')
    plt.plot(range(total_budget), pvalue_external, color='green', label='External Model p-value')
    plt.xlabel('Iteration')
    plt.ylabel('p-value')
    plt.title(f'p-value of Internal and External Models on {args.dataset.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/pvalue_{args.dataset.upper()}.png')


    
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Define the maximum anchor size and total budget for the experiment
    max_anchor = 16000
    total_budget = 200
    
    # Train the SMBO model and get the results
    result_performances, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external = train_smbo(args, max_anchor=max_anchor, total_budget=total_budget)
    
    # Plot the results
    plotting(total_budget, result_performances, spearman_corr_internal, pvalue_internal, spearman_corr_external, pvalue_external)