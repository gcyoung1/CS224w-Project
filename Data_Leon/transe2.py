import snap
import numpy as np
import math
import random
import create_tneanet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from os import listdir, mkdir
from os.path import join, isdir, isfile

import time
def d(triple, embeddings):
    head, relation, tail = triple
    pred = - embeddings[head] + embeddings[tail]
    return np.dot(embeddings[head], embeddings[tail]) + np.dot(embeddings[relation], pred)

class TranseDSET(Dataset):
    def __init__(self, graph, k):
        self.edge_embeddings = {}
        self.triplets = []
        self.G = graph
        width = 6/math.sqrt(k)
        for edge in self.G.Edges():
            init = torch.rand(k) * 2 * width - width
            self.edge_embeddings[graph.GetStrAttrDatE(edge, "kind")] = init/np.sqrt(init.dot(init))
            self.triplets.append((edge.GetSrcNId(), graph.GetStrAttrDatE(edge, "kind"), edge.GetDstNId()))
        self.node_embeddings = torch.rand(self.G.GetNodes(), k) * 2 * width - width
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        h, l, t = self.triplets[idx]
        ct = self.G.GetRndNId()
        return self.node_embeddings[h], self.edge_embeddings[l], self.node_embeddings[t], self.node_embeddings[ct]



def transe(graph, k, margin, batch_size, learning_rate, epochs, embeddings = None):
    #create triplets, initialize embeddings
    dontinit = True
    if embeddings is None:
        dontinit = False
    triplets = []


    t0 = time.time()
    for e in range(epochs):
        perm = np.random.permutation(len(triplets))
        print(perm)
        num_batches = int(perm.shape[0]/batch_size)
        for i in range(num_batches):
            if i%128 == 0:
                print("Epoch %d, batch %d of %d, t: %.2fs" %(e, i, num_batches, time.time() - t0))
            batch = [triplets[perm[j]] for j in range(i*batch_size, (i+1)*batch_size)]
            pairs = []
            corruptTail = False
            #create batch with good and fraud triples
            for trip in batch:
                if random.randint(0,1):
                    corruptTail = True
                    newTrip = (trip[0], trip[1], graph.GetRndNId(Rnd))
                else:
                    newTrip = (graph.GetRndNId(Rnd), trip[1], trip[2])
                pairs.append((trip,newTrip))

            #compute loss
            gradients = {}
            for pair in pairs:
                #take gradient of
                if margin+d(pair[0], embeddings)-d(pair[1], embeddings) > 0:
                    #gradient of L
                    if pair[0][1] not in gradients:
                        gradients[pair[0][1]] = embeddings[pair[0][2]] - embeddings[pair[0][0]] - (embeddings[pair[1][2]] - embeddings[pair[1][0]])
                    else:
                        gradients[pair[0][1]] = gradients[pair[0][1]] + (embeddings[pair[0][2]] - embeddings[pair[0][0]] - (embeddings[pair[1][2]] - embeddings[pair[1][0]]))

                    #gradient of head
                    temp = embeddings[pair[0][2]] - embeddings[pair[0][1]]
                    if corruptTail: temp = temp - (embeddings[pair[1][2]] - embeddings[pair[0][1]])
                    if pair[0][0] not in gradients:
                        gradients[pair[0][0]] =  temp
                    else:
                        gradients[pair[0][0]] = gradients[pair[0][0]] + temp

                    #gradient of tail
                    temp = embeddings[pair[0][0]] + embeddings[pair[0][1]]
                    if not corruptTail: temp = temp - (embeddings[pair[1][0]] + embeddings[pair[0][1]])
                    if pair[0][2] not in gradients:
                        gradients[pair[0][2]] =  temp
                    else:
                        gradients[pair[0][2]] = gradients[pair[0][2]] + temp

                    #gradient of corrupted head
                    if not corruptTail:
                        if pair[1][0] not in gradients:
                            gradients[pair[1][0]] =  embeddings[pair[0][2]] - embeddings[pair[0][1]]
                        else:
                            gradients[pair[1][0]] = gradients[pair[1][0]] + embeddings[pair[0][2]] - embeddings[pair[0][1]]

                    #gradient of corrupted tail
                    if corruptTail:
                        if pair[1][2] not in gradients:
                            gradients[pair[1][2]] =  embeddings[pair[0][0]] + embeddings[pair[0][1]]
                        else:
                            gradients[pair[1][2]] = gradients[pair[1][2]] + embeddings[pair[0][0]] + embeddings[pair[0][1]]

            for key in gradients:
                embeddings[key] = embeddings[key] - learning_rate*gradients[key]

            # normalize entity embeddings
        for node in graph.Nodes():
            embeddings[node.GetId()] = embeddings[node.GetId()] / np.sqrt(
                embeddings[node.GetId()].dot(embeddings[node.GetId()]))
    print("Time elapsed: %f" % (time.time() - t0))
    return embeddings

import create_tneanet as ct
def milestone():
    G = ct.load_tneanet()
    # find nodes of interest
    players = {32496: 'Allan McGregor',
        168308: 'Danny Wilson',
        32618: 'Kevin Thomson',
        148483: 'Marian Kello',
        37307: 'Marius Zaliukas',
        46359: 'Eggert Jonsson',}
    h_team = (8548, "Rangers")
    a_team = (9860, "Heart of Midlothian")
    match = 658980
    country = (19694, "Scotland")
    nodeIs = {}
    for ni in G.Nodes():
        kind = G.GetStrAttrDatN(ni, "kind")
        if kind == 'player'\
            and G.GetStrAttrDatN(ni, "playerName") in players.values():
                nodeIs[G.GetStrAttrDatN(ni, "playerName")] = ni.GetId()
        elif kind == 'country' and G.GetStrAttrDatN(ni, "countryName") == country[1]:
            nodeIs[country[1]] = ni.GetId()
        elif kind == 'match' and G.GetIntAttrDatN(ni, "matchId") == match:
                nodeIs[match] = ni.GetId()
        elif kind == 'team':
            if G.GetStrAttrDatN(ni, "teamName") == h_team[1]:
                nodeIs[h_team[1]] = ni.GetId()
            elif G.GetStrAttrDatN(ni, "teamName") == a_team[1]:
                nodeIs[a_team[1]] = ni.GetId()
    stopAt = [0, 1, 3, 10]
    savedEmbs = {}
    embeddings = transe(G, 2, 1, 128, 0.1, 0)
    for i in range(stopAt[-1]):
        if i in stopAt:
            savedEmbs[i] = {}
            for (k, v) in nodeIs.items():
                savedEmbs[i][k] = embeddings[v]
        embeddings = transe(G, 2, 1, 128, 0.1, 1)

    savedEmbs[10] = {}
    for (k, v) in nodeIs.items():
        savedEmbs[10][k] = embeddings[v]
    return nodeIs, G, savedEmbs

import networkx as nx
import matplotlib.pyplot as plt
def showEmbeddings(embDict):
    # x = [v[0] for v in embDict.values()]
    # y = [v[1] for v in embDict.values()]
    # plt.scatter(x, y, s=400)
    # for k, (x, y) in embDict.items():
    #     plt.text(x, y, k)
    # plt.show()
    G_1 = nx.Graph()
    tempedgelist = [[0,9], [1,9], [2,9], [0,1], [0,2], [1,3], [1,4], [1,6], [2,5], [2,7], [2,8]]
    G_1.add_edges_from(tempedgelist)
    n_nodes = 10
    pos = {}
    x = [v[0] for v in embDict.values()]
    y = [v[1] for v in embDict.values()]
    txt = [s for s in embDict.keys()]
    for i in range(n_nodes):
        pos[i] = (x[i], y[i])
        plt.text(x[i], y[i], txt[i])
    nx.draw(G_1, pos, edge_labels=True)
    plt.show()




# print(transe(test, 2, 1, 1, 0.01, 50))