import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import s_gd2


def draw_graph(G, pos, drawEdges=True, node_size=30, node_color='#1a1a1a', edge_color='gray', edge_width=0.5, figsize=(12, 8), padding=0.1, title=None, with_labels=True, font_size=8, font_color='black', colors=None, savepath=None):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Draw edges
    if drawEdges:
        for (u, v) in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            line = plt.Line2D([x1, x2], [y1, y2], lw=edge_width,
                              color=edge_color, alpha=0.7, zorder=1)
            ax.add_line(line)

    # Draw nodes
    x = [pos[node][0] for node in G.nodes()]
    y = [pos[node][1] for node in G.nodes()]
    if colors == None:
        ax.scatter(x, y, s=node_size, c=node_color, zorder=2)
    else:
        ax.scatter(x, y, s=node_size, c=colors, zorder=2)

    # Calculate bounding box with padding
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Set plot limits with padding
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add labels
    if with_labels:
        labels = {node: str(node) for node in G.nodes()}
        # Adjust this value to change label distance from node
        x_offset = x_range * 0.02
        y_offset = y_range * 0.01

        for node in G.nodes():
            node_x, node_y = pos[node]
            label = labels[node]

            # Calculate label position
            label_x = node_x - x_offset
            label_y = node_y + y_offset

            ax.text(label_x, label_y, label, fontsize=font_size, color=font_color,
                    ha='center', va='center', zorder=1,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1))

    # Add legend
    if colors is not None and set(colors) == {'blue', 'red'}:
        blue_patch = mpatches.Patch(color='blue', label='Male (or Uknown)')
        red_patch = mpatches.Patch(color='red', label='Female')
        plt.legend(handles=[blue_patch, red_patch], loc='upper right')

    if title == None:
        plt.title("Graph Layout")
    else:
        plt.title(f'{title}')

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
        print(f'>Fig. saved in {savepath}')
        plt.close()
    else:
        plt.show()
        plt.close()


def sgd_2(G, weighted=False):
    I = []
    J = []
    V = []

    for u, v, data in G.edges(data=True):
        I.append(u)
        J.append(v)

        if weighted:
            V.append(data['weight'])

    if weighted:
        x = s_gd2.layout(I, J, V=V, t_max=200)
    else:
        x = s_gd2.layout(I, J, t_max=200)

    X = {}

    for i, d in enumerate(x):
        X[i] = d

    return X


def coloring_gender(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('gender', 0) == "male":
                color_list.append("blue")
            else:
                color_list.append("red")
                red_node_found = True

        if red_node_found:
            return color_list

def coloring_obesity(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('status', 0) == "obese" or G.nodes[node].get('status', 0) == "overweight":
                color_list.append("blue")
            else:
                color_list.append("red")
                red_node_found = True

        if red_node_found:
            return color_list

def coloring_standard(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('Gender', 0) == 0 or G.nodes[node].get('Gender', 0) == 1:
                color_list.append("blue")
            else:
                color_list.append("red")
                red_node_found = True

        if red_node_found:
            return color_list


def printAVN_gender():

    for index in range(0, 24):
        filename = f'graph_spa_500_{index}'
        filepath = f'./AVN/{filename}.graphml'
        G = nx.read_graphml(filepath)
        G = nx.convert_node_labels_to_integers(G)

        colors = {}
        colors = coloring_gender(G)

        pos = sgd_2(G)

        savepath = f'./AVN/output/gender/{filename}.png'
        draw_graph(G, pos, title=filename, with_labels=False,
                   colors=colors, savepath=savepath)


def printAVN_obesity():
    for index in range(0, 24):
        filename = f'graph_spa_500_{index}'
        filepath = f'./AVN/{filename}.graphml'
        G = nx.read_graphml(filepath)
        G = nx.convert_node_labels_to_integers(G)

        colors = {}
        colors = coloring_obesity(G)

        pos = sgd_2(G)

        savepath = f'./AVN/output/obesity/{filename}.png'
        draw_graph(G, pos, title=filename, with_labels=False,
                   colors=colors, savepath=savepath)


def printDrug_normal():
    filename = f'DrugNet_filter'
    filepath = f'./DrugNet/{filename}.graphml'
    G = nx.read_graphml(filepath)
    G = nx.convert_node_labels_to_integers(G)

    colors = {}
    colors = coloring_standard(G)

    pos = sgd_2(G)

    savepath = f'./DrugNet/output/{filename}.png'
    draw_graph(G, pos, title=filename, with_labels=False,
               colors=colors, savepath=savepath)


if __name__ == '__main__':
    # printAVN_gender()
    #printAVN_obesity()
    printDrug_normal()
