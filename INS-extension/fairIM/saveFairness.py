import networkx as nx
import numpy as np
import pickle
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator
import math


def multi_to_set(f, n=None):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    if n == None:
        n = len(g)

    def f_set(S):
        return f(indicator(S, n))
    return f_set


def valoracle_to_single(f, i):
    def f_single(x):
        return f(x, 1000)[i]
    return f_single


budget = 25
print('Budget: {}'.format(budget))


# whether fair influence share is calculated by forming a subgraph consisting
# only of nodes in a given subgroup -- leave this to True for the setting in
# the paper
succession = True

# what method to use to solve the inner maxmin LP. This can be either 'md' to
# use stochastic saddlepoint mirror descent, or 'gurobi' to solve the LP explicitly
# the the gurobi solver (requires an installation and license).
# MD is better asymptotically for large networks but may require many iterations
# and is typically slower than gurobi for small/medium sized problems. You can
# tune the stepsize/batch size/number of iterations for MD by editing algorithms.py
solver = 'md'

# attribute to examine fairness wrt
# attributes = ['birthsex', 'gender', 'sexori', 'race']
attributes = ['region', 'ethnicity', 'age', 'gender', 'status']


# network -> attribute -> n_runs * n_values
gr_values = {}
# network -> attribute -> n_runs * n_values
group_size = {}
# algorithm -> network -> attribute -> n_runs * n_values
alg_values = {}

num_runs = 1
algorithms = ['Greedy', 'GR', 'MaxMin-Size']
for alg in algorithms:
    alg_values[alg] = {}

num_graphs = 1
graphnames = ['spa_500_4']
print(graphnames)

for graphname in graphnames:

    g = pickle.load(open('networks/graph_{}.pickle'.format(graphname), 'rb'))

    # remove nodes without demographic information
    if 'spa' not in graphname:
        to_remove = []
        for v in g.nodes():
            if 'race' not in g.nodes[v]:
                to_remove.append(v)
        g.remove_nodes_from(to_remove)

    # propagation probability for the ICM
    p = 0.1
    for u, v in g.edges():
        g[u][v]['p'] = p

    g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

    gr_values[graphname] = {}
    group_size[graphname] = {}
    for alg in algorithms:
        alg_values[alg][graphname] = {}
    for attribute in attributes:
        # assign a unique numeric value for nodes who left the attribute blank
        nvalues = len(np.unique([g.nodes[v][attribute] for v in g.nodes()]))
        if 'spa' not in graphname:
            for v in g.nodes():
                if np.isnan(g.nodes[v][attribute]):
                    g.nodes[v][attribute] = nvalues
        nvalues = len(np.unique([g.nodes[v][attribute] for v in g.nodes()]))
        gr_values[graphname][attribute] = np.zeros((num_runs, nvalues))
        group_size[graphname][attribute] = np.zeros((num_runs, nvalues))
        for alg in algorithms:
            alg_values[alg][graphname][attribute] = np.zeros(
                (num_runs, nvalues))

    fair_vals_attr = np.zeros((num_runs, len(attributes)))
    greedy_vals_attr = np.zeros((num_runs, len(attributes)))
    pof = np.zeros((num_runs, len(attributes)))

    include_total = False

    for run in range(num_runs):
        print(graphname, run)
        live_graphs = sample_live_icm(g, 1000)

        group_indicator = np.ones((len(g.nodes()), 1))

        val_oracle = make_multilinear_objective_samples_group(
            live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))
        grad_oracle = make_multilinear_gradient_group(
            live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))

        def f_multi(x):
            return val_oracle(x, 1000).sum()

        # f_multi = make_multilinear_objective_samples(live_graphs, list(g.nodes()), list(g.nodes()), np.ones(len(g)))
        f_set = multi_to_set(f_multi)

        # find overall optimal solution
        S, obj = greedy(list(range(len(g))), budget, f_set)

        for attr_idx, attribute in enumerate(attributes):
            # all values taken by this attribute
            values = np.unique([g.nodes[v][attribute] for v in g.nodes()])

            values = np.unique([g.nodes[v][attribute] for v in g.nodes()])
            nodes_attr = {}
            for vidx, val in enumerate(values):
                nodes_attr[val] = [v for v in g.nodes() if g.nodes[v]
                                   [attribute] == val]
                group_size[graphname][attribute][run,
                                                 vidx] = len(nodes_attr[val])

            opt_succession = {}
            if succession:
                for vidx, val in enumerate(values):
                    h = nx.subgraph(g, nodes_attr[val])
                    h = nx.convert_node_labels_to_integers(h)
                    live_graphs_h = sample_live_icm(h, 1000)
                    group_indicator = np.ones((len(h.nodes()), 1))
                    val_oracle = multi_to_set(valoracle_to_single(make_multilinear_objective_samples_group(
                        live_graphs_h, group_indicator,  list(h.nodes()), list(h.nodes()), np.ones(len(h))), 0), len(h))
                    S_succession, opt_succession[val] = greedy(list(h.nodes()), math.ceil(
                        len(nodes_attr[val])/len(g) * budget), val_oracle)

            if include_total:
                group_indicator = np.zeros((len(g.nodes()), len(values)+1))
                for val_idx, val in enumerate(values):
                    group_indicator[nodes_attr[val], val_idx] = 1
                group_indicator[:, -1] = 1
            else:
                group_indicator = np.zeros((len(g.nodes()), len(values)))
                for val_idx, val in enumerate(values):
                    group_indicator[nodes_attr[val], val_idx] = 1

            val_oracle = make_multilinear_objective_samples_group(
                live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))
            grad_oracle = make_multilinear_gradient_group(
                live_graphs, group_indicator,  list(g.nodes()), list(g.nodes()), np.ones(len(g)))

            # build an objective function for each subgroup
            f_attr = {}
            f_multi_attr = {}
            for vidx, val in enumerate(values):
                nodes_attr[val] = [v for v in g.nodes() if g.nodes[v]
                                   [attribute] == val]
                f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
                f_attr[val] = multi_to_set(f_multi_attr[val])

            # get the best seed set for nodes of each subgroup
            S_attr = {}
            opt_attr = {}
            if not succession:
                for val in values:
                    S_attr[val], opt_attr[val] = greedy(list(range(len(g))), int(
                        len(nodes_attr[val])/len(g) * budget), f_attr[val])
            if succession:
                opt_attr = opt_succession
            all_opt = np.array([opt_attr[val] for val in values])
            gr_values[graphname][attribute][run] = all_opt

            # Calculate influence spread or other metrics using S
            xg = np.zeros(len(g.nodes))
            xg[list(S)] = 1
            greedy_vals = val_oracle(xg, 1000)

            save_data = {
                'selected_nodes': list(S),
                'live_graphs': live_graphs,
                'group_indicator': group_indicator,
                'graph_nodes': list(g.nodes()),
                'graph_edges': list(g.edges(data=True)),
                'attributes': attributes,
                'attribute_values': values,
                'budget': budget,
                'algorithms': algorithms,
                'graphname': graphname
            }

            # Save the data to a file using pickle
            with open('greedy_results.pkl', 'wb') as f:
                pickle.dump(save_data, f)

            print("Data saved successfully.")

            # Compute and Save Fair_x
            threshold = 5
            targets = [opt_attr[val] for val in values]
            if include_total:
                targets.append(1.025*obj)
            targets = np.array(targets)
            fair_x = algo(grad_oracle, val_oracle, threshold, budget,
                          group_indicator, np.array(targets), 100, solver)[1:]
            fair_x = fair_x.mean(axis=0)

            # Calculate the influence spread using the fair_x vector
            all_fair_vals = val_oracle(fair_x, 1000)

            # Save the fair algorithm results
            fair_save_data = {
                'fair_x': fair_x,
                'live_graphs': live_graphs,
                'group_indicator': group_indicator,
                'graph_nodes': list(g.nodes()),
                'graph_edges': list(g.edges(data=True)),
                'attributes': attributes,
                'attribute_values': values,
                'budget': budget,
                'global_spread_fair': all_fair_vals.sum(),
                'group_spreads_fair': all_fair_vals
            }

            # Save the data to a file using pickle
            with open('fair_results.pkl', 'wb') as f:
                pickle.dump(fair_save_data, f)

            print("Fair algorithm data saved successfully.")

            # After computing the minmax_x
            grad_oracle_normalized = make_normalized(
                grad_oracle, group_size[graphname][attribute][run])
            val_oracle_normalized = make_normalized(
                val_oracle, group_size[graphname][attribute][run])
            minmax_x = maxmin_algo(grad_oracle_normalized, val_oracle_normalized, threshold, budget,
                                   group_indicator, 20, 10, 0.05, solver)
            minmax_x = minmax_x.mean(axis=0)

            # Calculate the influence spread using the minmax_x vector
            all_minmax_vals = val_oracle(minmax_x, 1000)

            # Save the MinMax algorithm results
            minmax_save_data = {
                'minmax_x': minmax_x,
                'live_graphs': live_graphs,
                'group_indicator': group_indicator,
                'graph_nodes': list(g.nodes()),
                'graph_edges': list(g.edges(data=True)),
                'attributes': attributes,
                'attribute_values': values,
                'budget': budget,
                'global_spread_minmax': all_minmax_vals.sum(),
                'group_spreads_minmax': all_minmax_vals
            }

            # Save the data to a file using pickle
            with open('minmax_results.pkl', 'wb') as f:
                pickle.dump(minmax_save_data, f)

            print("MinMax algorithm data saved successfully.")

            break
