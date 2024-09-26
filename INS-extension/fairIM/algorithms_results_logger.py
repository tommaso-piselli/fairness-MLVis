import networkx as nx
import numpy as np
import json
import pickle
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from utils import greedy
from algorithms import algo, maxmin_algo, make_normalized


def add_edge_probabilities(g, default_p=0.1):
    for u, v in g.edges():
        if 'p' not in g[u][v]:
            g[u][v]['p'] = default_p
    return g


def run_algorithms(g, budget, attribute='region', solver='md', num_iterations=100, threshold=5):
    # Add edge probabilities if they don't exist
    g = add_edge_probabilities(g)

    live_graphs = sample_live_icm(g, 1000)
    group_indicator = np.ones((len(g.nodes()), 1))

    # Objective and gradient oracles
    val_oracle = make_multilinear_objective_samples_group(
        live_graphs, group_indicator, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
    grad_oracle = make_multilinear_gradient_group(
        live_graphs, group_indicator, list(g.nodes()), list(g.nodes()), np.ones(len(g)))

    # Greedy algorithm
    def greedy_algorithm(g, budget):
        def f_multi(x): return val_oracle(x, 1000).sum()
        def f_set(S): return f_multi(
            np.array([1 if i in S else 0 for i in range(len(g))]))
        S, obj = greedy(list(range(len(g))), budget, f_set)
        return list(S)

    # Fair algorithm (GR)
    def fair_algorithm(g, budget):
        fair_x = algo(grad_oracle, val_oracle, threshold, budget,
                      group_indicator, np.array([1.025*budget]), num_iterations, solver)
        fair_x = fair_x.mean(axis=0)
        return list(np.where(fair_x > 0.5)[0])  # convert to a list of integers

    # MaxMin-Size algorithm
    def maxmin_algorithm(g, budget):
        grad_oracle_normalized = make_normalized(grad_oracle, np.ones(len(g)))
        val_oracle_normalized = make_normalized(val_oracle, np.ones(len(g)))
        minmax_x = maxmin_algo(grad_oracle_normalized, val_oracle_normalized,
                               threshold, budget, group_indicator, num_iterations, 10, 0.05, solver)
        minmax_x = minmax_x.mean(axis=0)
        # convert to a list of integers
        return list(np.where(minmax_x > 0.5)[0])

    algorithms = {
        'Greedy': greedy_algorithm,
        'GR': fair_algorithm,
        'MaxMin-Size': maxmin_algorithm
    }

    results = {}
    for alg_name, alg_func in algorithms.items():
        selected_nodes = alg_func(g, budget)
        node_info = [(int(node), g.nodes[node][attribute])
                     for node in selected_nodes]

        # Calculate spread
        x = np.zeros(len(g))
        x[selected_nodes] = 1
        spread = float(val_oracle(x, 1000)[0])

        results[alg_name] = {
            'nodes': node_info,
            'spread': spread
        }

    return results


def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    graph_file = 'networks/graph_spa_500_4.pickle'
    with open(graph_file, 'rb') as f:
        g = pickle.load(f)

    budget = 25
    results = run_algorithms(g, budget)
    save_results(results, 'algorithm_results_with_spread.json')


if __name__ == "__main__":
    main()
