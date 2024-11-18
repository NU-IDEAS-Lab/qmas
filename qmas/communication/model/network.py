import networkx as nx

class CommunicationNetwork(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return str(self.nodes) + "\n" + str(self.edges)