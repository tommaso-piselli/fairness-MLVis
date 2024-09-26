import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

filepath = f'./DrugNet/CSV'
filename = f'DRUGNET'
df = pd.read_csv(f'{filepath}/{filename}.csv', index_col=0)

# Step 2: Convert the DataFrame to a NumPy array
adj_matrix = df.to_numpy()

# Step 3: Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)
nx.to_undirected(G)

edges = [edge for edge in sorted(G.edges())]
H = nx.Graph()
H.add_nodes_from(G)
H.add_edges_from(edges)

# Aggiungere gli attributi dal csv
attr_df = pd.read_csv(f'{filepath}/DRUGATTR.csv', index_col=0)


# Step 6: Clean up the column names and index, and shift IDs down by 1
attr_df.columns = attr_df.columns.str.replace('*', '')
attr_df.index = attr_df.index.astype(int) - 1

# Step 7: Add attributes to each node in the graph
idx = 0
for node, row in attr_df.iterrows():
    H.nodes[idx]['Ethnicity'] = row['Ethnicity']
    H.nodes[idx]['Gender'] = row['Gender']
    H.nodes[idx]['HasTie'] = row['HasTie']
    idx = idx + 1

# Scrivere attributi su txt
attribute_counts = {
    'Ethnicity': defaultdict(int),
    'Gender': defaultdict(int),
    'HasTie': defaultdict(int)
}

# Iterate through all nodes and their attributes
for node, data in H.nodes(data=True):
    for attr, value in data.items():
        if attr in attribute_counts:
            attribute_counts[attr][value] += 1

# Write the dictionaries to a text file
# with open(f'{filepath}/{filename}_summary.txt', 'w') as f:
#     n = H.number_of_nodes()
#     m = H.number_of_edges()
#     f.write(f'Graph: {filename}\nNodes: {n}\nEdges: {m}\nDensity: {m/n}\n')
#     f.write("\n")
#     for attr, counts in attribute_counts.items():
#         f.write(f"{attr.capitalize()}:\n")
#         for value, count in sorted(counts.items()):
#             f.write(f"  {value}: {count}\n")
#         f.write("\n")  # Add a blank line for readability

# print(f'Attribute counts have been written to {filename}_summary.txt')

# savepath = f'./DrugNet/graphml/DrugNet.graphml'
# nx.write_graphml_lxml(H, savepath)
# print(f'Written to {savepath}')

nodes_to_remove = [185, 126, 138, 141, 87, 119, 11, 222,
                   38, 216, 180, 177, 58, 163, 193, 203, 237, 144, 150]
filter_nodes = [(node_id, H.nodes[node_id])
                for node_id in H.nodes() if H.nodes[node_id].get('HasTie', 0) != 0 and node_id not in nodes_to_remove]


G_filter = nx.Graph()
G_filter.add_nodes_from(filter_nodes)
filter_edges = [edge for edge in edges if edge[0]
                in G_filter.nodes() and edge[1] in G_filter.nodes()]
G_filter.add_edges_from(filter_edges)

# Remappo gli id in modo tale da avere valori consecutivi e senG_filtera buchi
node_ids = [node[0] for node in filter_nodes]
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(node_ids))}
nodes_remap = [(id_mapping[node[0]], node[1]) for node in filter_nodes]
edges_remap = [(id_mapping[edge[0]], id_mapping[edge[1]])
               for edge in filter_edges]

G_filter_remap = nx.Graph()
G_filter_remap.add_nodes_from(nodes_remap)
G_filter_remap.add_edges_from(edges_remap)


# Scrivere attributi su txt
attribute_counts_filter = {
    'Ethnicity': defaultdict(int),
    'Gender': defaultdict(int),
    'HasTie': defaultdict(int)
}

# Iterate through all nodes and their attributes
for node, data in G_filter_remap.nodes(data=True):
    for attr, value in data.items():
        if attr in attribute_counts_filter:
            attribute_counts_filter[attr][value] += 1

# Write the dictionaries to a text file
with open(f'{filepath}/{filename}_filter_summary.txt', 'w') as f:
    n = G_filter_remap.number_of_nodes()
    m = G_filter_remap.number_of_edges()
    f.write(f'Graph: {filename}\nNodes: {n}\nEdges: {m}\nDensity: {m/n}\n')
    f.write("\n")
    for attr, counts in attribute_counts_filter.items():
        f.write(f"{attr.capitalize()}:\n")
        for value, count in sorted(counts.items()):
            f.write(f"  {value}: {count}\n")
        f.write("\n")  # Add a blank line for readability

print(f'Attribute counts have been written to {filename}_summary.txt')
savepath = f'./DrugNet/graphml/DrugNet_filter.graphml'
nx.write_graphml_lxml(G_filter_remap, savepath)
print(f'Written to {savepath}')

#  Ripulire il Dataset -> Basta leggere dall'altro CSV se ha ties o no
