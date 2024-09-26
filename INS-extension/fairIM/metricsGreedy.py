import pickle
import numpy as np
import networkx as nx
from icm import make_multilinear_objective_samples_group

# Specify the graph name you want to load
graphname = 'spa_500_4'

# Load the graph from the pickle file
g = pickle.load(open(f'networks/graph_{graphname}.pickle', 'rb'))

# If needed, remove nodes without demographic information (example code)
if 'spa' not in graphname:
    to_remove = []
    for v in g.nodes():
        if 'race' not in g.nodes[v]:
            to_remove.append(v)
    g.remove_nodes_from(to_remove)

# Set the propagation probability for the ICM
p = 0.1
for u, v in g.edges():
    g[u][v]['p'] = p

# Convert node labels to integers and maintain the 'pid' attribute
g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

# Extract the saved data
with open('greedy_results.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

selected_nodes = loaded_data['selected_nodes']
live_graphs = loaded_data['live_graphs']
group_indicator = loaded_data['group_indicator']
attributes = loaded_data['attributes']
values = loaded_data['attribute_values']
budget = loaded_data['budget']

# Recreate val_oracle
val_oracle = make_multilinear_objective_samples_group(
    live_graphs, group_indicator, list(g.nodes()), list(
        g.nodes()), np.ones(len(g.nodes()))
)

# Function to compute and print metrics


def compute_metrics(S, val_oracle, g, attributes, values, budget):
    xg = np.zeros(len(g.nodes))
    xg[S] = 1
    greedy_vals = val_oracle(xg, 1000)

    # Print the shape and content of greedy_vals to understand its structure
    print(f"Shape of greedy_vals: {greedy_vals.shape}")
    # print(f"Content of greedy_vals: {greedy_vals}")

    # If greedy_vals has two elements, assume they correspond to two groups
    if len(greedy_vals) == 2:
        print(f"Global Spread - Greedy: {greedy_vals.sum()}")
        for i, val in enumerate(values):
            print(f"Group {val} Spread - Greedy: {greedy_vals[i]}")
    else:
        raise ValueError(
            f"Unexpected length of greedy_vals: {len(greedy_vals)}. Expected 2.")


# Compute metrics for the original set of selected nodes
print(f'Selected Nodes: {selected_nodes}')
compute_metrics(selected_nodes, val_oracle, g, attributes, values, budget)

# Implement node swapping and recompute metrics


def swap_and_evaluate(graph, selected_nodes, val_oracle, attributes):
    node_list = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}

    for node in selected_nodes:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if graph.nodes[neighbor][attributes[0]] != graph.nodes[node][attributes[0]]:
                print(f"\nSwapping {node} with {neighbor}")

                new_nodes = list(selected_nodes)
                new_nodes.remove(node)
                new_nodes.append(neighbor)

                # Convert new_nodes from node IDs to indices if needed
                compute_metrics(new_nodes, val_oracle, graph,
                                attributes, values, budget)


# Perform swap and evaluate
swap_and_evaluate(g, selected_nodes, val_oracle, attributes)
