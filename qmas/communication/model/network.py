import networkx as nx

class CommunicationNetwork(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_id = 0

    def add_node(self, *args, **kwargs):
        super().add_node(self._node_id, *args, **kwargs)
        self._node_id += 1

    def add_edge(self, *args, **kwargs):
        super().add_edge(*args, **kwargs)

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_edge(self, node_id1, node_id2):
        return self.edges[node_id1, node_id2]

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_neighbors(self, node_id):
        return self.neighbors(node_id)

    def get_degree(self, node_id):
        return self.degree(node_id)

    def get_node_id(self):
        return self._node_id

    def set_node_id(self, node_id):
        self._node_id = node_id

    def remove_node(self, node_id):
        super().remove_node(node_id)

    def remove_edge(self, node_id1, node_id2):
        super().remove_edge(node_id1, node_id2)

    def clear(self):
        self.clear()

    def __str__(self):
        return str(self.nodes) + "\n" + str(self.edges)