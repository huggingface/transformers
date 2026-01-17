"""
=====================
Krackhardt Centrality
=====================

Centrality measures of Krackhardt social network.
"""

import matplotlib.pyplot as plt
import networkx as nx

G = nx.krackhardt_kite_graph()

print("Betweenness")
b = nx.betweenness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {b[v]:.3f}")

print("Degree centrality")
d = nx.degree_centrality(G)
for v in G.nodes():
    print(f"{v:2} {d[v]:.3f}")

print("Closeness centrality")
c = nx.closeness_centrality(G)
for v in G.nodes():
    print(f"{v:2} {c[v]:.3f}")

pos = nx.spring_layout(G, seed=367)  # Seed layout for reproducibility
nx.draw(G, pos)
plt.show()
