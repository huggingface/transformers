"""
====================
Custom Node Position
====================

Draw a graph with node(s) located at user-defined positions.

When a position is set by the user, the other nodes can still be neatly organised in a layout.
"""

import networkx as nx
import numpy as np

G = nx.path_graph(20)  # An example graph
center_node = 5  # Or any other node to be in the center
edge_nodes = set(G) - {center_node}
# Ensures the nodes around the circle are evenly distributed
pos = nx.circular_layout(G.subgraph(edge_nodes))
pos[center_node] = np.array([0, 0])  # manually specify node position
nx.draw(G, pos, with_labels=True)
