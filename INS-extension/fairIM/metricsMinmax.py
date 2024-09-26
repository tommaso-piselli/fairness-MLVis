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

# Load the saved MinMax algorithm data
with open('minmax_results.pkl', 'rb') as f:
    minmax_data = pickle.load(f)

minmax_x = minmax_data['minmax_x']
group_indicator = minmax_data['group_indicator']
attributes = minmax_data['attributes']
values = minmax_data['attribute_values']
budget = minmax_data['budget']

# Recreate val_oracle
val_oracle = make_multilinear_objective_samples_group(
    minmax_data['live_graphs'], group_indicator, list(
        g.nodes()), list(g.nodes()), np.ones(len(g.nodes()))
)

# Take the top k nodes with the highest percentages from minmax_x
top_k_minmax = np.argsort(minmax_x)[-budget:][::-1]
print(top_k_minmax)

# Function to compute and print metrics for MinMax algorithm


def compute_minmax_metrics(S, val_oracle, g, attributes, values, budget):
    xg = np.zeros(len(minmax_x))
    xg[S] = 1
    minmax_vals = val_oracle(xg, 1000)

    # Print the shape and content of minmax_vals to understand its structure
    print(f"Shape of minmax_vals: {minmax_vals.shape}")
    print(f"Content of minmax_vals: {minmax_vals}")

    # If minmax_vals has two elements, assume they correspond to two groups
    if len(minmax_vals) == 2:
        print(f"Global Spread - MinMax: {minmax_vals.sum()}")
        for i, val in enumerate(values):
            print(f"Group {val} Spread - MinMax: {minmax_vals[i]}")
    else:
        raise ValueError(
            f"Unexpected length of minmax_vals: {len(minmax_vals)}. Expected 2.")


# Compute metrics for the original set of selected nodes in minmax_x
compute_minmax_metrics(top_k_minmax.tolist(), val_oracle,
                       g, attributes, values, budget)

# Implement node swapping and recompute metrics


def swap_and_evaluate_minmax(graph, selected_nodes, minmax_x, val_oracle, attributes):
    node_list = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}

    for node in selected_nodes:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if graph.nodes[neighbor][attributes[0]] != graph.nodes[node][attributes[0]]:
                print(f"\nSwapping {node} with {neighbor} in MinMax")

                # Swap the percentages in the minmax_x vector
                minmax_x[node_to_index[neighbor]], minmax_x[node_to_index[node]
                                                            ] = minmax_x[node_to_index[node]], minmax_x[node_to_index[neighbor]]

                # Recompute the metrics with the swapped nodes
                new_minmax_vals = val_oracle(minmax_x, 1000)
                new_global_spread_minmax = new_minmax_vals.sum()
                print(
                    f"Global Spread after swap - MinMax: {new_global_spread_minmax}")

                # Directly print the new group spreads
                if len(new_minmax_vals) == 2:
                    for i, val in enumerate(values):
                        print(
                            f"Group {val} Spread - MinMax after swap: {new_minmax_vals[i]}")
                else:
                    raise ValueError(
                        "Unexpected number of group spreads in new_minmax_vals")

                # Revert the swap
                minmax_x[node_to_index[neighbor]], minmax_x[node_to_index[node]
                                                            ] = minmax_x[node_to_index[node]], minmax_x[node_to_index[neighbor]]


print("\n--- Start Swap and Evaluate for MinMax Algorithm ---")
swap_and_evaluate_minmax(g, top_k_minmax.tolist(),
                         minmax_x, val_oracle, attributes)
print("--- End Swap and Evaluate for MinMax Algorithm ---\n")
