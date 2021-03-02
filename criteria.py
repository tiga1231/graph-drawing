from pynndescent import NNDescent
import lovasz_losses as L


import utils
import torch
from torch import nn
import numpy as np
import networkx as nx

import random



def neighborhood_preseration(pos, G, adj, k2i, i2k, 
                             n_roots=2, depth_limit=2, 
                             neg_sample_rate=0.5, 
                             device='cpu'):
    
    pos_samples = []
    for _ in range(n_roots):
        root = i2k[random.randint(0, len(G)-1)]
        G_sub = nx.bfs_tree(G, root, depth_limit=depth_limit)
        pos_samples += [k2i[n] for n in G_sub.nodes]
    pos_samples = sorted(set(pos_samples))
    
    n_neg_samples = int(neg_sample_rate * len(pos_samples))
    neg_samples = [random.randint(0, len(G)-1) for _ in range(n_neg_samples)] ##negative samples
    
    samples = sorted(set(
        pos_samples + neg_samples
    )) ##remove duplicates
    
    
    pos = pos[samples,:]
    adj = adj[samples,:][:, samples]
    
    n,m = pos.shape
    x = pos
    
    ## k_dist
    degrees = adj.sum(1).numpy().astype(np.int64)
    max_degree = degrees.max().item()
    n_neighbors = max(2, min(max_degree+1, n))

    n_trees = min(64, 5 + int(round((n) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(n))))
    
    knn_search_index = NNDescent(
        x.detach().numpy(),
        n_neighbors=n_neighbors,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    
    kmax = knn_dists.shape[1]-1
    k_dist = np.array([
        ( knn_dists[i,min(kmax, k)] + knn_dists[i,min(kmax, k+1)] ) / 2 
        for i,k in enumerate(degrees)
    ])

    ## pdist
    x0 = x.repeat(1, n).view(-1,m)
    x1 = x.repeat(n, 1)
    pdist = nn.PairwiseDistance()(x0, x1).view(n, n)


    ## loss 
    pred = torch.from_numpy(k_dist.astype(np.float32)).view(-1,1) - pdist
    target = adj + torch.eye(adj.shape[0], device=device)
    loss = L.lovasz_hinge(pred, target)
    return loss




def edge_uniformity(pos, G, k2i, n_samples=None):
    n,m = pos.shape[0], pos.shape[1]

    if n_samples is not None:
        edges = random.sample(G.edges, n_samples)
    else:
        edges = G.edges

    sourceIndices, targetIndices = zip(*[ [k2i[e0], k2i[e1]] for e0,e1 in edges])
    source = pos[sourceIndices,:]
    target = pos[targetIndices,:]
    edgeLengths = (source-target).norm(dim=1) 
    eu = edgeLengths.std()
    return eu




def stress(pos, D, W, n_samples=None):
    n,m = pos.shape[0], pos.shape[1]
    if n_samples is not None:
        i0 = np.random.choice(n, n_samples)
        i1 = np.random.choice(n, n_samples)
        x0 = pos[i0,:]
        x1 = pos[i1,:]
        D = torch.tensor([D[i,j] for i, j in zip(i0, i1)])
        W = torch.tensor([W[i,j] for i, j in zip(i0, i1)])
    else:
        x0 = X.repeat(1, n).view(-1,m)
        x1 = X.repeat(n, 1)
        D = D.view(-1)
        W = W.view(-1)
    pdist = nn.PairwiseDistance()(x0, x1)
    diff = pdist-D
#     wbound = (1/4 * diff.abs().min()).item()
# #     print(W.max(), wbound)
#     W.clamp_(0, wbound)
    return (W*(diff)**2).sum()




