import pickle
import numpy as np
import networkx as nx
from icm import make_multilinear_objective_samples_group

# Specify the graph name you want to load
graphname = 'spa_500_4'

# Load the graph from the pickle file
g = pickle.load(open(f'networks/graph_{graphname}.pickle', 'rb'))

# Set the propagation probability for the ICM
p = 0.1
for u, v in g.edges():
    g[u][v]['p'] = p

# Convert node labels to integers and maintain the 'pid' attribute
g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

# Load the saved fair algorithm data
with open('fair_results.pkl', 'rb') as f:
    fair_data = pickle.load(f)

fair_x = fair_data['fair_x']
group_indicator = fair_data['group_indicator']
attributes = fair_data['attributes']
values = fair_data['attribute_values']
budget = fair_data['budget']

# Recreate val_oracle
val_oracle = make_multilinear_objective_samples_group(
    fair_data['live_graphs'], group_indicator, list(
        g.nodes()), list(g.nodes()), np.ones(len(g.nodes()))
)

# Take the top k nodes with the highest percentages from fair_x
top_k_fair = np.argsort(fair_x)[-budget:][::-1]
print(top_k_fair)

# Function to compute and print metrics for Fair algorithm


def compute_fair_metrics(S, val_oracle, g, attributes, values, budget):
    xg = np.zeros(len(fair_x))
    xg[S] = 1
    fair_vals = val_oracle(xg, 1000)

    # Print the shape and content of fair_vals to understand its structure
    print(f"Shape of fair_vals: {fair_vals.shape}")
    print(f"Content of fair_vals: {fair_vals}")

    # If fair_vals has two elements, assume they correspond to two groups
    if len(fair_vals) == 2:
        print(f"Global Spread - Fair: {fair_vals.sum()}")
        for i, val in enumerate(values):
            print(f"Group {val} Spread - Fair: {fair_vals[i]}")
    else:
        raise ValueError(
            f"Unexpected length of fair_vals: {len(fair_vals)}. Expected 2.")


# Compute metrics for the original set of selected nodes in fair_x
compute_fair_metrics(top_k_fair.tolist(), val_oracle,
                     g, attributes, values, budget)

# Implement node swapping and recompute metrics


def swap_and_evaluate_fair(graph, selected_nodes, fair_x, val_oracle, attributes):
    node_list = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}

    for node in selected_nodes:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if graph.nodes[neighbor][attributes[0]] != graph.nodes[node][attributes[0]]:
                print(f"\nSwapping {node} with {neighbor} in Fair")

                # Swap the percentages in the fair_x vector
                fair_x[node_to_index[neighbor]], fair_x[node_to_index[node]
                                                        ] = fair_x[node_to_index[node]], fair_x[node_to_index[neighbor]]

                # Recompute the metrics with the swapped nodes
                new_fair_vals = val_oracle(fair_x, 1000)
                new_global_spread_fair = new_fair_vals.sum()
                print(
                    f"Global Spread after swap - Fair: {new_global_spread_fair}")

                # Directly print the new group spreads
                if len(new_fair_vals) == 2:
                    for i, val in enumerate(values):
                        print(
                            f"Group {val} Spread - Fair after swap: {new_fair_vals[i]}")
                else:
                    raise ValueError(
                        "Unexpected number of group spreads in new_fair_vals")

                # Revert the swap
                fair_x[node_to_index[neighbor]], fair_x[node_to_index[node]
                                                        ] = fair_x[node_to_index[node]], fair_x[node_to_index[neighbor]]


print("\n--- Start Swap and Evaluate for Fair Algorithm ---")
swap_and_evaluate_fair(g, top_k_fair.tolist(), fair_x, val_oracle, attributes)
print("--- End Swap and Evaluate for Fair Algorithm ---\n")
