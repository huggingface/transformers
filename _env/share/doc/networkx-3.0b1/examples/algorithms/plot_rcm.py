"""
======================
Reverse Cuthill--McKee
======================

Cuthill-McKee ordering of matrices

The reverse Cuthill--McKee algorithm gives a sparse matrix ordering that
reduces the matrix bandwidth.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# build low-bandwidth matrix
G = nx.grid_2d_graph(3, 3)
rcm = list(nx.utils.reverse_cuthill_mckee_ordering(G))
print("ordering", rcm)

print("unordered Laplacian matrix")
A = nx.laplacian_matrix(G)
x, y = np.nonzero(A)
# print(f"lower bandwidth: {(y - x).max()}")
# print(f"upper bandwidth: {(x - y).max()}")
print(f"bandwidth: {(y - x).max() + (x - y).max() + 1}")
print(A)

B = nx.laplacian_matrix(G, nodelist=rcm)
print("low-bandwidth Laplacian matrix")
x, y = np.nonzero(B)
# print(f"lower bandwidth: {(y - x).max()}")
# print(f"upper bandwidth: {(x - y).max()}")
print(f"bandwidth: {(y - x).max() + (x - y).max() + 1}")
print(B)

sns.heatmap(B.todense(), cbar=False, square=True, linewidths=0.5, annot=True)
plt.show()
