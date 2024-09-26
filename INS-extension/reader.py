import networkx as nx
import scipy.io as io
import os
import torch
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix


class Graph(object):

    def __init__(self, name, filepath):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.G = nx.Graph()
        self.name = name
        self.filepath = filepath
        self.startPos = None
        self.pos = None
        self.D = None
        self.W = None
        self.feature = None
        self.dataset = None

    def create_folders(self):
        '''
        Creates folders
        Returns: folders for the figures at the filepath fp that we specify
        '''
        if self.feature is not None and self.dataset is not None:
            fp = f'output/figs/{self.dataset}/{self.feature}/{self.name}'
        else:
            fp = f'output/figs/{self.name}'

        if not os.path.exists(fp):
            os.mkdir(fp)
            print(f'Folder created @ {fp}')
        else:
            print(f'> Folder {fp} already present')

    def shortest_path(self):
        """
        Computes the shortest paths between all pairs of nodes in a graph.

        Args:
            G (networkx.Graph): The graph.

        Returns:
            np.array: A 2D array representing the shortest paths between all pairs of nodes.
        """

        # credits to: https://github.com/tiga1231/graph-drawing/blob/sgd/utils/utils.py

        k2i = {k: i for i, k in enumerate(self.G.nodes)}
        edge_indices = np.array([(k2i[n0], k2i[n1])
                                for (n0, n1) in self.G.edges])
        row_indices = edge_indices[:, 0]
        col_indices = edge_indices[:, 1]
        adj_data = np.ones(len(edge_indices))
        adj_sparse = csr_matrix((
            adj_data,
            (row_indices, col_indices)
        ), shape=(len(self.G), len(self.G)), dtype=np.float32)

        D = csgraph.shortest_path(adj_sparse, directed=False, unweighted=True)
        return D

    def compute_paths_and_weights(self):
        self.D = self.shortest_path()
        self.D = torch.from_numpy(self.D).to(self.device)
        self.W = 1 / (self.D ** 2 + 1e-6).to(self.device)
        print('> Shortest paths (D) and Weights (W) computed!')

    def load_graphml(self, feature=None, dataset=None):
        '''
        Loads a graph in a .graphml format
        '''

        self.G = nx.read_graphml(self.filepath)
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        print(f'Graph: {self.name} loaded')
        print(f'Vertices: {nx.number_of_nodes(self.G)}')
        print(f'Edges: {nx.number_of_edges(self.G)}')

        # Computes shortest paths and weights
        self.compute_paths_and_weights()

        if feature is not None:
            self.feature = feature

        if dataset is not None:
            self.dataset = dataset

        # Create Folders for the figures
        self.create_folders()

    def load_mat(self):
        """
        Loads a graph from a MATLAB .mat file in the SuiteSparse Matrix Collection format.

        Args:
            fn (str, optional): Path to the .mat file. Defaults to 'data/lesmis.mat'.

        Returns:
            networkx.Graph: The loaded graph.
        """

        # credits to : https://github.com/jxz12/s_gd2/blob/master/jupyter/main.ipynb
        # load the data from the SuiteSparse Matrix Collection format
        # https://www.cise.ufl.edu/research/sparse/matrices/
        # Note: pay attention of the .mat format (i.e. the version of Matlab in which the file was saved)

        mat_data = io.loadmat(self.filepath)
        adj = mat_data['Problem']['A'][0][0]

        # Load
        self.G = nx.convert_matrix.from_numpy_array(adj.toarray())
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        print(f'Graph: {self.name} loaded')
        print(f'Vertices: {nx.number_of_nodes(self.G)}')
        print(f'Edges: {nx.number_of_edges(self.G)}')

        # Computes shortest paths and weights
        self.compute_paths_and_weights()

        # Create Folders for the figures
        self.create_folders()

    def load_nx(self, key):
        Graphs = [
            ('karate', nx.karate_club_graph()),
            ('path-5', nx.path_graph(5)),
            ('path-10', nx.path_graph(10)),
            ('cycle-10', nx.cycle_graph(10)),
            ('grid-5-5', nx.grid_graph(dim=[5, 5])),
            ('grid-10-6', nx.grid_graph(dim=[10, 6])),
            ('tree-2-3', nx.balanced_tree(2, 3)),
            ('tree-2-4', nx.balanced_tree(2, 4)),
            ('tree-2-5', nx.balanced_tree(2, 5)),
            ('tree-2-6', nx.balanced_tree(2, 6)),
            ('k-5', nx.complete_graph(5)),
            ('k-20', nx.complete_graph(20)),
            ('bipartite-graph-3-3', nx.complete_bipartite_graph(3, 3)),
            ('bipartite-graph-5-5', nx.complete_bipartite_graph(5, 5)),
            ('dodecahedron', nx.dodecahedral_graph()),
            ('cube', nx.hypercube_graph(3)),
        ]

        self.name = Graphs[key][0]
        self.G = Graphs[key][1]

        need_mapping = self.name == 'grid-5-5' or self.name == 'grid-10-6' or self.name == 'dodecahedron' or self.name == 'cube'
        if need_mapping:
            mapping = {node: i for i, node in enumerate(self.G.nodes())}
            self.G = nx.relabel_nodes(self.G, mapping)

        print(f'Graph: {self.name} loaded')
        print(f'Vertices: {nx.number_of_nodes(self.G)}')
        print(f'Edges: {nx.number_of_edges(self.G)}')

        # Computes shortest paths and weights
        self.compute_paths_and_weights()

        # Create Folders for the figures
        self.create_folders()

    def set_pos(self):
        startPos = (len(self.G.nodes) ** 0.5) * \
            torch.randn(len(self.G.nodes), 2, device=self.device)
        print('> Nodes positions correctly initialize')
        self.startPos = startPos

    def init_positions(self, device=None):
        """
        Computes initial node positions for graph drawing, considering shortest paths and fairness.

        Args:
            pos (torch.Tensor): Initial node positions.
            G (networkx.Graph): The graph.
            device (torch.device): The device to use for tensor operations.

        Returns:
            tuple: A tuple containing:
                - pos (torch.Tensor): Updated node positions.
                - D (torch.Tensor): Shortest path distances between all pairs of nodes.
                - W (torch.Tensor): Weights for fairness objective.
        """
        device = self.device
        self.pos = self.startPos
        self.pos.requires_grad_(True)

        # self.D = torch.from_numpy(self.D).to(device)
        # self.W = 1 / (self.D**2 + 1e-6).to(device)
