import pandas as pd
from ConfigSpace import ConfigurationSpace
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import math
from dataset_handler import get_dataframes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='lcdb')
    parser.add_argument('--budget', type=int, default=100000)

    return parser.parse_args()

def successive_halving(args, B=100000, arms=1000):
    dataset = get_dataframes()[args.dataset]
    config_space = ConfigurationSpace.from_json(args.config_space_file)
    
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(dataset)
    
    S = [dict(conf) for conf in config_space.sample_configuration(arms)]
    halving_steps = math.ceil(math.log2(arms))
    bandit_performance = {i: [] for i in range(len(S))}

    for i in range(halving_steps):
        budget = B / (len(S) * halving_steps)
        for bandit in S:
            bandit["anchor_size"] = budget
            bandit["score"] = surrogate_model.predict(bandit)
        
        for idx, bandit in enumerate(S):
            bandit_performance[idx].append(bandit["score"])
        
        S.sort(key=lambda x: x["score"])
        S = S[:math.ceil(len(S) / 2)]

    best_bandit_id = min(bandit_performance, key=lambda k: bandit_performance[k][-1])
    best_performance = min(bandit_performance[best_bandit_id])

    return {
        'best_bandit_id': best_bandit_id,
        'best_performance_array': bandit_performance[best_bandit_id],
        'best_performance': best_performance,
        'num_configs': arms,
    }

# Main function to run the steps
def main(args):
    run1 = successive_halving(args, B=100000, arms=1000)
    print(run1)
    run2 = successive_halving(args, B=100000, arms=4000)
    run3 = successive_halving(args, B=100000, arms=8000)
    
    # Plot the performance of the best performing bandit for both runs
    plt.plot(range(len(run1['best_performance_array'])), run1['best_performance_array'], marker='o', label='Best configuration out of 1000, run 1', color='orange', alpha=0.3)
    plt.plot(range(len(run2['best_performance_array'])), run2['best_performance_array'], marker='o', label='Best configuration out of 4000, run 2', color='orange')
    plt.plot(range(len(run3['best_performance_array'])), run3['best_performance_array'], marker='o', label='Best configuration out of 8000, run 3', color='red', alpha=0.7)
    
    # Draw horizontal dashed lines for the best performance of both runs
    plt.axhline(y=run1['best_performance'], color='black', linestyle='--', label=f'Best performance, run 1: {run1["best_performance"]:.2f}', alpha=0.3)
    plt.axhline(y=run2['best_performance'], color='black', linestyle='--', label=f'Best performance, run 2: {run2["best_performance"]:.2f}')
    plt.axhline(y=run3['best_performance'], color='red', linestyle='--', label=f'Best performance, run 3: {run3["best_performance"]:.2f}', alpha=0.7)
    plt.xlabel('Halving Step')
    plt.ylabel('Score')
    plot_title = f'Performance of Best Configuration in Successive Halving on {args.dataset.upper()}'
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    # plt.show()
    save_path = f'./plots/SUCCESSIVE_HALVING_PERFORMANCE_{args.dataset.upper()}.png'
    plt.savefig(save_path)

if __name__ == "__main__":
    main(parse_args())
