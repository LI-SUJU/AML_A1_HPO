import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from dataset_handler import get_dataframes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=8000)
    parser.add_argument('--num_iterations', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='lcdb')

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    # df = pd.read_csv(args.configurations_performance_file)
    df = get_dataframes()[args.dataset]
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    all_results = []
    best_performance = float('inf')

    for run_idx in range(3):
        random_search = RandomSearch(config_space)
        results = {
            'random_search': [1.0]
        }

        for idx in range(args.num_iterations):
            theta_new = dict(random_search.select_configuration())
            theta_new['anchor_size'] = args.max_anchor_size
            performance = surrogate_model.predict(theta_new)
            # ensure to only record improvements
            results['random_search'].append(min(results['random_search'][-1], performance))
            random_search.update_runs((theta_new, performance))

        all_results.append(results['random_search'])
        best_performance = min(best_performance, min(results['random_search']))

    for run_idx, run_results in enumerate(all_results):
        plt.plot(range(len(run_results)), run_results, label=f'Run {run_idx + 1}')

    # grid
    # plt.grid(True)
    # more space for y lable
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    plt.axhline(y=best_performance, color='black', linestyle='--', label=f'Best performance: {best_performance}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    title = f'Random Search on {args.dataset.upper()}'
    plt.title(title)
    plt.legend()
    # plt.show()
    save_path = f'./plots/random_search_{args.dataset}.png'
    plt.savefig(save_path)


if __name__ == '__main__':
    run(parse_args())
