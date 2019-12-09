# import numpy as np
# import matplotlib.pyplot as plt

# data = [[a,b] for a in range(0,2) for b in range(0,2)]
# x = [a[0] for a in data]
# y = [a[1] for a in data]
# plt.scatter(x,y, s=400)
# for x in data:  
#     plt.text(x[0], x[1], "Hi")
# plt.show()
# labels = ['point{0}'.format(i) for i in range(len(data))]

# plt.subplots_adjust(bottom = 0.1)
# plt.scatter(
#     data[:, 0], data[:, 1], marker='o', c=data[:, 2], s=data[:, 3] * 1500,
#     cmap=plt.get_cmap('Spectral'))

# for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x, y), xytext=(-20, 20),
#         textcoords='offset points', ha='right', va='bottom',
#         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import random

G_1 = nx.Graph()
tempedgelist =  [[0, 2], [0, 3], [1, 2], [1, 4], [5, 3]]
G_1.add_edges_from(tempedgelist)

n_nodes = 6
pos = {}
for i in range(n_nodes):
    pos[i] = (i,i)
    plt.text(i,i,"HI")
#nx.draw(G_1, pos, edge_labels=True)
nx.draw_networkx_edge_labels(G_1,pos,edge_labels={(0,2):'02',\
(0,3):'03',(1,2):'12'},font_color='red')
plt.show()