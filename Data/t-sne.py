import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#embeddings_matrix is 2d array of embeddings whose columns are dimensions and whose rows are separate embeddings
#type_list is a list of types for each embedding (e.g. ['player', 'team', 'player', 'country'])
#labels_list is a list of labels for each embedding (e.g. ["France", "Messi"])
def helper(embeddings_matrix, type_list, label_list):
    new_embeddings = TSNE().fit_transform(embeddings_matrix)
    color_dict = {"player": "red", "country": "blue", "team": "green", "match": "yellow"}
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)

    xmin = min(new_embeddings[:,0])
    xmax = max(new_embeddings[:,0])
    rad = (xmax - xmin)/20
    ax.set_xlim([xmin-rad, xmax+rad])
    ax.set_ylim([min(new_embeddings[:,1])-rad, max(new_embeddings[:,1])+rad])
    for i, embedding in enumerate(new_embeddings):
        circle = plt.Circle((embedding[0], embedding[1]), radius = rad, color=color_dict[type_list[i]])
        ax.add_patch(circle)
        label = ax.annotate(label_list[i], xy = (embedding[0], embedding[1]), fontsize="10", va = "center", ha = "center")
    plt.show()

#embeddings is the dict of embeddings
#node_ids is the node ids you wish to plot
#G is the graph
def plot_embeddings(embeddings, node_ids, G):
    label_dict = {"team": "teamName", "country": "countryName", "player": "playerName", "match": "stageId"}
    embeddings_matrix, type_list, label_list = [], [], []
    for id in node_ids:
        embeddings_matrix.append(embeddings[id])
        kind = G.GetStrAttrDatN(id, "kind")
        type_list.append(kind)
        label_list.append(G.GetStrAttrDatN(id, label_dict[kind]))
    helper(embeddings_matrix, type_list, label_list)

embeddings = [[0,1,5,3,-5,6], [1,2, 0, -3, 3, -1], [3,2, -2,-1, 6,2]]
type_list = ["player", "country", "player"]
label_list = ["Messi", "Brazil", "Ronaldo"]
helper(embeddings, type_list, label_list)