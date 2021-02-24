## model criterion
import utils

import torch
from torch import nn
import numpy as np

import random

# def getSamples(G, n, includeNeighbors=True):
#     seedNodes = random.sample(G.nodes, n)
#     samples = set(seedNodes)
#     if includeNeighbors:
#         for i in seedNodes:
#             neighbors = set(G.neighbors(i))
#             samples.update(neighbors)
#     samples = list(samples)
#     return samples


# def stress_update_ij(X, D, W, i, j, base_lr=1):
#     with torch.no_grad():
#         diff = X[i] - X[j]
#         norm = diff.norm()
#         r = (norm-D[i,j]) / 2 / (norm+0.001) * diff
#         final_lr = min(0.99, base_lr * W[i,j])
#         grad_xi = r
#         grad_xj = -r
#         X[i].data -= final_lr * grad_xi
#         X[j].data -= final_lr * grad_xj


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


