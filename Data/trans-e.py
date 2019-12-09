import snap
import numpy as np
import math
import random
import create_tneanet
Rnd = snap.TRnd(37349)
Rnd.Randomize()

realDeal = create_tneanet.load_tneanet()

test = snap.TNEANet.New()
test.AddNode(2)
test.AddNode(3)
test.AddNode(1)
test.AddEdge(1,2,1)
test.AddEdge(1,3,2)


def d(triple, embeddings):
    head, relation, tail = triple
    pred = [t[1]-t[0] for t in zip(embeddings[head], embeddings[tail])]
    return np.dot(head, tail) + np.dot(relation, pred)

def transe(graph, k, margin, batch_size, learning_rate, epochs):
    #create triplets, initialize embeddings
    embeddings = {}
    triplets = []
    for edge in graph.Edges():
        #what is the attribute?
        init = np.random.uniform(-6/math.sqrt(k), 6/math.sqrt(k), k)
        embeddings[graph.GetStrAttrDatE(edge, "kind")] = init/np.sqrt(init.dot(init))
        triplets.append((edge.GetSrcNId(), graph.GetStrAttrDatE(edge, "kind"), edge.GetDstNId()))
    for node in graph.Nodes():
        embeddings[node.GetId()] = np.random.uniform(-6/math.sqrt(k), 6/math.sqrt(k), k)

    for _ in range(epochs):
        #normalize entity embeddings
        for node in graph.Nodes():
            embeddings[node.GetId()] = embeddings[node.GetId()]/np.sqrt(embeddings[node.GetId()].dot(embeddings[node.GetId()]))
        batch = np.sample(triplets, batch_size)
        gradients = {}
        #create batch with good and fraud triples
        for trip in batch:
            corruptTail = False
            if random.randint(0,1):
                corruptTail = True
                newTrip = (trip[0], trip[1], graph.GetRndNId(Rnd))
            else:
                newTrip = (graph.GetRndNId(Rnd), trip[1], trip[2])

            #compute loss
            if margin+d(trip, embeddings)-d(newTrip, embeddings) > 0:
                #gradient of L
                if trip[1] not in gradients:
                    gradients[trip[1]] = embeddings[trip[2]] - embeddings[trip[0]] - (embeddings[newTrip[2]] - embeddings[newTrip[0]])
                else:
                    gradients[trip[1]] = gradients[trip[1]] + (embeddings[trip[2]] - embeddings[trip[0]] - (embeddings[newTrip[2]] - embeddings[newTrip[0]]))
                
                #gradient of head
                temp = embeddings[trip[2]] - embeddings[trip[1]]
                if corruptTail: temp = temp - (embeddings[newTrip[2]] - embeddings[trip[1]])
                if trip[0] not in gradients:
                    gradients[trip[0]] =  temp
                else:
                    gradients[trip[0]] = gradients[trip[0]] + temp

                #gradient of tail
                temp = embeddings[trip[0]] + embeddings[trip[1]]
                if not corruptTail: temp = temp - (embeddings[newTrip[0]] + embeddings[trip[1]])
                if trip[2] not in gradients:
                    gradients[trip[2]] =  temp
                else:
                    gradients[trip[2]] = gradients[trip[2]] + temp

                #gradient of corrupted head
                if not corruptTail:
                    if newTrip[0] not in gradients:
                        gradients[newTrip[0]] =  embeddings[trip[2]] - embeddings[trip[1]]
                    else:
                        gradients[newTrip[0]] = gradients[newTrip[0]] + embeddings[trip[2]] - embeddings[trip[1]]

                #gradient of corrupted tail
                if corruptTail:
                    if newTrip[2] not in gradients:
                        gradients[newTrip[2]] =  embeddings[trip[0]] + embeddings[trip[1]]
                    else:
                        gradients[newTrip[2]] = gradients[newTrip[2]] + embeddings[trip[0]] + embeddings[trip[1]]

        for key in embeddings:
            embeddings[key] = embeddings[key] + learning_rate*gradients[key]
    return embeddings

print(transe(realDeal, 2, 1, 1, 0.01, 50))