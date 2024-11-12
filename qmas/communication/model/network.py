import networkx as nx

class CommunicationNetwork(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_edge_qos(self, node_id1, node_id2):
        # TODO: Implement QoS! Perhaps we should add a QoS object to each edge as an attribute, then do the calculation when that attribute is accessed.
        return None

    def __str__(self):
        return str(self.nodes) + "\n" + str(self.edges)