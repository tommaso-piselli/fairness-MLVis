import pickle
import networkx as nx
from collections import defaultdict

# Load the graph from the pickle file

for index in range (0,24):
    filepath = r'data\preprocessing\summary\AVN'
    filename = f'graph_spa_500_{index}'

    with open(rf'{filepath}\{filename}.pickle', 'rb') as f:
        G = pickle.load(f)

    # Initialize a dictionary to store dictionaries for each attribute
    attribute_counts = {
        'region': defaultdict(int),
        'ethnicity': defaultdict(int),
        'age': defaultdict(int),
        'gender': defaultdict(int),
        'status': defaultdict(int)
    }

    # Iterate through all nodes and their attributes
    for node, data in G.nodes(data=True):
        for attr, value in data.items():
            if attr in attribute_counts:
                attribute_counts[attr][value] += 1

    # Write the dictionaries to a text file
    with open(rf'{filepath}\summary\{filename}_summary.txt', 'w') as f:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        f.write(f'Graph: {filename}\nNodes: {n}\nEdges: {m}\nDensity: {m/n}\n')
        f.write("\n") 
        for attr, counts in attribute_counts.items():
            f.write(f"{attr.capitalize()}:\n")
            for value, count in sorted(counts.items()):
                f.write(f"  {value}: {count}\n")
            f.write("\n")  # Add a blank line for readability

    print(f'Attribute counts have been written to {filename}_summary.txt')
    
    # savepath = f'AVN\summary\{filename}.graphml'
    # nx.write_graphml_lxml(G, savepath)
    # print(f'\t>Written to {savepath}')