import random
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

'''
    | VARIABLES
'''

rt = None
bt = None
nr = 0
nb = 0

NODE_SIZE = 15
EDGE_COLOR = '#03adfc'
EDGE_WIDTH = 0.8
EDGE_ALPHA = 0.3


def count_colors(n, device, G):
    """
    Counts the number of nodes of each color in the graph.

    Args:
        n (int): Number of nodes in the graph.
        device (torch.device): The device to use for tensor operations.
        G (networkx.Graph): The graph.

    Updates global variables `rt`, `bt`, `nr`, and `nb`.
    """

    global nr, nb, rt, bt
    rt = torch.zeros(n*n).to(device)
    bt = torch.zeros(n*n).to(device)
    for i in range(n*n):
        index_u = i // n
        cu = G.nodes[index_u]['color']
        if cu == "red":
            rt[i] = 1
            nr += 1
        else:
            bt[i] = 1
            nb += 1
    nr = nr / n
    nb = nb / n


def nodeColoring(G, percentage):
    """
    Assigns random colors to nodes in the graph such that a certain percentage
    of nodes are colored red.

    Args:
        G (networkx.Graph): The graph.
        percentage (float): The percentage of nodes to color red.

    Returns:
        dict, list: A dictionary mapping nodes to colors and a list of colors.
    """

    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            node_color = random.randint(1, 100)
            if node_color >= percentage:
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def stress_per_node(pos, D, W):
    """
    Calculates the stress metric for each node in the graph.

    Args:
        pos (torch.Tensor): Node positions.
        D (torch.Tensor): Shortest path distances between all pairs of nodes.
        W (torch.Tensor): Weights for fairness objective.

    Returns:
        torch.Tensor: Stress value for each node.
    """

    n, m = pos.shape[0], pos.shape[1]

    x0 = pos.repeat(1, n).view(-1, m)
    x1 = pos.repeat(n, 1)
    D = D.view(-1)
    W = W.view(-1)
    pdist = nn.PairwiseDistance()(x0, x1)

    stress_per_node = W*(pdist-D)**2

    # Reshape to per-node stress
    stress_per_node = stress_per_node.view(n, n).sum(dim=1) / (n-1)

    return stress_per_node


def nodeColoring_stress(G, percentage, pos, D, W):
    """
    Assigns colors to nodes in the graph such that a certain percentage
    of nodes are colored red based on stress values.

    Args:
        G (networkx.Graph): The graph.
        percentage (float): The percentage of nodes to color red.
        pos (torch.Tensor): Node positions.
        D (torch.Tensor): Shortest path distances between all pairs of nodes.
        W (torch.Tensor): Weights for fairness objective.

    Returns:
        dict: A dictionary mapping node names to colors.
    """

    # Compute stress values per node
    stress_values = stress_per_node(pos, D, W)

    # Determine threshold for top percentage% most stressed nodes
    threshold = np.percentile(
        stress_values.cpu().detach().numpy(), 100 - percentage)

    # Coloring
    node_colors = {}
    red_node_found = False

    for node, stress in zip(G.nodes, stress_values.cpu().detach().numpy()):
        if stress >= threshold:
            node_colors[node] = "red"
            red_node_found = True

        else:
            node_colors[node] = "blue"

    if red_node_found == False:
        for node in G.nodes:
            node_colors[node] = "blue"
        node_colors[0] = "red"

    return node_colors


def stress(graph):
    """
    Calculates the stress metric for graph drawing, which measures the
    difference between the euclidean distance and the shortest path distance between nodes.

    Args:
        pos (torch.Tensor): Node positions.
        D (torch.Tensor): Shortest path distances between all pairs of nodes.
        W (torch.Tensor): Weights for fairness objective.

    Returns:
        torch.Tensor: The average stress value.
    """

    n, m = graph.pos.shape[0], graph.pos.shape[1]

    x0 = graph.pos.repeat(1, n).view(-1, m)
    x1 = graph.pos.repeat(n, 1)
    D = graph.D.view(-1)
    W = graph.W.view(-1)
    pdist = nn.PairwiseDistance()(x0, x1)

    res = W*(pdist-D)**2

    return res.mean()


def unfairness(graph):
    """
    Calculates the unfairness metric for graph drawing, which measures the
    difference in average stress between red and blue nodes.

    Args:
        pos (torch.Tensor): Node positions.
        D (torch.Tensor): Shortest path distances between all pairs of nodes.
        W (torch.Tensor): Weights for fairness objective.
        G (networkx.Graph): The graph.
        device (torch.device): The device to use for tensor operations.

    Returns:
        torch.Tensor: The squared difference in average stress between red and blue nodes.
    """

    n, m = graph.pos.shape[0], graph.pos.shape[1]

    x0 = graph.pos.repeat(1, n).view(-1, m)
    x1 = graph.pos.repeat(n, 1)
    D = graph.D.view(-1)
    W = graph.W.view(-1)
    pdist = nn.PairwiseDistance()(x0, x1)
    res = W * (pdist - D) ** 2
    if nr == 0:
        reds = rt*res*0
    else:
        reds = rt*res*1/nr
    if nb == 0:
        blues = bt*res*0
    else:
        blues = bt*res*1/nb

    sumt = (reds - blues).mean()
    return sumt.pow(2)


def plot_graph(graph, ax=None, node_size=NODE_SIZE, edge_width=EDGE_WIDTH, colors=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    pos = graph.pos.cpu().detach()

    for (n0, n1) in graph.G.edges:
        ax.plot([pos[n0, 0], pos[n1, 0]], [pos[n0, 1], pos[n1, 1]],
                alpha=EDGE_ALPHA, linewidth=EDGE_WIDTH, color=EDGE_COLOR, zorder=1)

    for color in colors:
        ax.scatter(pos[color, 0], pos[color, 1], s=node_size,
                   color=colors[color], edgecolors='white', zorder=2)

    ax.grid(True, zorder=0, alpha=0.15)
    ax.set_facecolor('#FFFFFF')

    if title:
        ax.set_title(title)


def plot_graph_variant(graph, node_ids=None, ax=None, node_size=NODE_SIZE, edge_width=EDGE_WIDTH, colors=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    pos = graph.pos.cpu().detach()

    # Define edge colors
    BLUE_EDGE_COLOR = 'blue'
    RED_EDGE_COLOR = 'red'
    MIXED_EDGE_COLOR = 'green'

    HIGHLIGHT_NODE = 'orange'
    STANDARD_NODE = 'white'

    # Dictionary for storing the nodes that need labels and their corresponding colors
    labeled_nodes = {}

    for (n0, n1) in graph.G.edges:
        # Determine edge color
        if colors[n0] == 'blue' and colors[n1] == 'blue':
            edge_color = BLUE_EDGE_COLOR
        elif colors[n0] == 'red' and colors[n1] == 'red':
            edge_color = RED_EDGE_COLOR
            # Add these nodes to labeled_nodes, ensuring no overwrites
            if n0 not in labeled_nodes:
                labeled_nodes[n0] = 'red'
            if n1 not in labeled_nodes:
                labeled_nodes[n1] = 'red'
        else:
            edge_color = MIXED_EDGE_COLOR

        ax.plot([pos[n0, 0], pos[n1, 0]], [pos[n0, 1], pos[n1, 1]],
                alpha=EDGE_ALPHA, linewidth=edge_width, color=edge_color, zorder=1)

    # Plot the nodes and labels
    for node in range(pos.shape[0]):
        if node in node_ids:
            labeled_nodes[node] = 'black'
            ax.scatter(pos[node, 0], pos[node, 1], s=node_size,
                       color=colors[node], edgecolors=HIGHLIGHT_NODE, zorder=2)
        else:
            ax.scatter(pos[node, 0], pos[node, 1], s=node_size,
                       color=colors[node], edgecolors=STANDARD_NODE, zorder=2)

        # Add labels to nodes in labeled_nodes
        # if node in labeled_nodes:
        #     ax.text(pos[node, 0]-0.1, pos[node, 1]-0.1, str(node),
        #             fontsize=3, ha='right', va='bottom', color=labeled_nodes[node], zorder=3)

    ax.grid(True, zorder=0, alpha=0.15)
    ax.set_facecolor('#FFFFFF')
    if title:
        ax.set_title(title)


def save_graph(graph, colors=None, savepath=None, ax=None, title=None):
    """
    Saves the plotted graph as an SVG image.

    Args:
        pos (torch.Tensor): Node positions.
        G (networkx.Graph): The graph.
        node_dic (dict, optional): Dictionary mapping nodes to colors. Defaults to None.
        save_path (str, optional): Path to save the image. Defaults to None.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes object.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    plot_graph(graph, ax=ax, colors=colors, title=title)
    plt.savefig(f'{savepath}_{title}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)


def save_graph_variant(graph, node_ids=None, colors=None, savepath=None, ax=None, title=None):
    """
    Saves the plotted graph as an SVG image.

    Args:
        pos (torch.Tensor): Node positions.
        G (networkx.Graph): The graph.
        node_dic (dict, optional): Dictionary mapping nodes to colors. Defaults to None.
        save_path (str, optional): Path to save the image. Defaults to None.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes object.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    plot_graph_variant(graph, node_ids=node_ids, ax=ax,
                       colors=colors, title=title)
    plt.savefig(f'{savepath}_{title}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

# def save_graph(graph, colors=None, savepath=None, title=None):

#     # with torch.no_grad():
#     #     pos_cpu = graph.pos.cpu().numpy()

#     draw_graph(graph, title=title, with_labels=False,
#                    colors=colors, savepath=f'{savepath}_{title}')


# New Experiments
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
    if colors is None:
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

    # Add labels only for nodes with edges that have different-colored endpoints
    if with_labels and colors is not None:
        labels = {node: str(node) for node in G.nodes()}
        # Adjust this value to change label distance from node
        x_offset = x_range * 0.02
        y_offset = y_range * 0.01

        nodes_to_label = set()  # Set to keep track of nodes to label

        for u, v in G.edges():
            # Check if the connected nodes have different colors
            if colors[u] != colors[v]:
                nodes_to_label.add(u)
                nodes_to_label.add(v)

        for node in nodes_to_label:
            node_x, node_y = pos[node]
            label = labels[node]

            # Calculate label position
            label_x = node_x - x_offset
            label_y = node_y + y_offset

            ax.text(label_x, label_y, label, fontsize=font_size, color=font_color,
                    ha='center', va='center', zorder=1,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1))

    # Save the plot if a path is provided
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()

    # Add legend
    # TODO: Cambiare la legenda
    # if colors is not None and set(colors) == {'blue', 'red'}:
    #     blue_patch = mpatches.Patch(color='blue', label='Obese or Overweight')
    #     red_patch = mpatches.Patch(color='red', label='Normal')
    #     plt.legend(handles=[blue_patch, red_patch], loc='upper right')

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

# New coloring


def coloring_AVN_gender(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('gender', 0) == "male":
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_AVN_status(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('status', 0) == "obese" or G.nodes[node].get('status', 0) == "overweight":
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_AVN_ethnicity(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('ethnicity', 0) != "white":
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_AVN_region(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('region', 0) != "lake_los_angeles":
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_AVN_age(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('age', 0) != "60-64" and G.nodes[node].get('age', 0) != "65+":
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_DrugNet_gender(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('Gender', 0) == 0 or G.nodes[node].get('Gender', 0) == 1:
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic


def coloring_DrugNet_ethnicity(G):
    while True:
        # Coloring
        node_dic = {}
        color_list = []
        red_node_found = False

        for node in G:
            if G.nodes[node].get('Ethnicity', 0) != 3:
                color_list.append("blue")
                node_dic[node] = "blue"
            else:
                color_list.append("red")
                node_dic[node] = "red"
                red_node_found = True

        if red_node_found:
            return node_dic
