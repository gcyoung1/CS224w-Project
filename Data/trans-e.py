import snap
import numpy as np
import math
import random
Rnd = snap.TRnd(42)
Rnd.Randomize()


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
        embeddings[graph.GetStrAttrDatE(edge, "kind")] = init/np.sqrt(x.dot(x))
        triplets.append((edge.GetSrcNId(), graph.GetStrAttrDatE(edge, Attr), edge.GetSrcNId()))
    for node in graph.Nodes():
        embeddings[node.GetId()] = np.random.uniform(-6/math.sqrt(k), 6/math.sqrt(k), k)

    for _ in range(epochs):
        #normalize entity embeddings
        for node in graph.Nodes():
            embeddings[node.GetId()] = embeddings[node.GetId()]/np.sqrt(embeddings[node.GetId()].dot(embeddings[node.GetId()]))
        batch = np.sample(triplets, batch_size)
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

        for key in embeddings:
            embeddings[key] = embeddings[key] + learning_rate*gradients[key]
    return embeddings

print(transe(test, 2, 1, 1, 0.01, 50))