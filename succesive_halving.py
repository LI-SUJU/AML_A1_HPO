import pandas as pd
import random
from ConfigSpace import ConfigurationSpace, read_and_write
from smbo import SequentialModelBasedOptimization
from typing import List, Tuple, Dict
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import math



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=25)

    return parser.parse_args()

# Main function to run the steps
def main(args):
    # File paths for dataset and config space
    dataset_path = args.configurations_performance_file
    config_path = args.config_space_file
    
    dataset = pd.read_csv(dataset_path)
    
    # Read the configuration space definition (JSON format assumed for ConfigSpace)
    config_space= ConfigurationSpace.from_json(config_path)

    # Train the surrogate model
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(dataset)
    
    B, arms = 100000, 1000
    S = [dict(conf) for conf in config_space.sample_configuration(arms)]

    halving_steps = math.ceil(math.log2(arms))
    bandit_performance = {i: [] for i in range(len(S))}
    # predictions_times = [0] * halving_steps

    for i in range(halving_steps):
        budget = B / (len(S) * halving_steps)
        for bandit in S:
            bandit["anchor_size"] = budget
            bandit["score"] = surrogate_model.predict(bandit)
            # predictions_times[i] = predictions_times[i] + 1
        
        for idx, bandit in enumerate(S):
            bandit_performance[idx].append(bandit["score"])
        
        S.sort(key=lambda x: x["score"])
        S = S[:math.ceil(len(S) / 2)]

    # Plot the best performing bandit
    best_bandit_id = min(bandit_performance, key=lambda k: bandit_performance[k][-1])
    best_performance = min(bandit_performance[best_bandit_id])
    
    plt.plot(range(len(bandit_performance[best_bandit_id])), bandit_performance[best_bandit_id], marker='o', label=f'Configuration {best_bandit_id}', color='orange')
    
    # Draw a horizontal dashed line for the best performance
    plt.axhline(y=best_performance, color='black', linestyle='--', label=f'Best Performance: {best_performance:.2f}')
    
    plt.xlabel('Halving Step')
    plt.ylabel('Score')
    plt.title('Performance of Best Configuration in Successive Halving on LCDB')
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig('successive_halving_performance_LCDB.png')

    # plot the performance of the best performing bandit over predictions_times
    # plt.plot(predictions_times, bandit_performance[best_bandit_id], marker='o', label=f'Bandit {best_bandit_id}', color='red')
    # plt.xlabel('Number of Predictions')
    # plt.ylabel('Score')
    # plt.title('Performance of Best Configuration in Successive Halving on LCDB')
    # plt.grid(True)
    # plt.legend()
    # # plt.show()
    # plt.savefig('successive_halving_performance_LCDB_over_predictions_times.png')


if __name__ == "__main__":
    main(parse_args())
