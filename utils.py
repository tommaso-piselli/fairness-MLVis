import random
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

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

def plot_graph(graph, ax=None, node_size=NODE_SIZE, edge_width=EDGE_WIDTH, node_dic=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    pos = graph.pos.cpu().detach()

    for (n0, n1) in graph.G.edges:
        ax.plot([pos[n0, 0], pos[n1, 0]], [pos[n0, 1], pos[n1, 1]],
                alpha=EDGE_ALPHA, linewidth=EDGE_WIDTH, color=EDGE_COLOR, zorder=1)

    for color in node_dic:
        ax.scatter(pos[color, 0], pos[color, 1], s=node_size,
                   color=node_dic[color], edgecolors='white', zorder=2)

    ax.grid(True, zorder=0, alpha=0.15)
    ax.set_facecolor('#FFFFFF')

    if title:
        ax.set_title(title)

def save_graph(graph, node_dic=None, save_path=None, ax=None, title=None):
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

    plot_graph(graph, ax=ax, node_dic=node_dic, title=title)
    plt.savefig(f'{save_path}_{title}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)