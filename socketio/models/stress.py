import numpy as np
import torch
import networkx as nx
import random


device = 'cpu'


def dict2tensor(d, fill=None):
    n = len(d.keys())
    k2i = {k:i for i,k in enumerate(sorted(d.keys()))}
    res = torch.zeros(len(d.keys()), len(d.keys()), device=device)
    for src_node, dst_nodes in d.items():
        for dst_node, distance in dst_nodes.items():
            if fill is not None:
                res[k2i[src_node],k2i[dst_node]] = fill
            else:
                res[k2i[src_node],k2i[dst_node]] = distance
    return res, k2i


def pairwise_distances(x, y=None, w=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is None:
        y = x
        y_t = y.t()
        y_norm = x_norm
    else:
        y_t = y.t()
        y_norm = (y**2).sum(1).view(1, -1)
        
    if w is not None:
        x = x * w    
        y = y * w    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)



class Stress():
    def __init__(self, graph):
        self.G = graph
        self.X = torch.rand(len(self.G.nodes), 2)
        self.D, k2i = dict2tensor(dict(nx.all_pairs_shortest_path_length(self.G)))
        self.W = 1 / (self.D**2+1)
        self.lr = 10;

    def update_ij(self, i=None, j=None):
        # self.lr *= 0.999
        if i is None:
            i = random.randint(0, len(self.G.nodes)-1)
        if j is None:
            j = random.randint(0, len(self.G.nodes)-1)
            while j==i:
                j = random.randint(0, len(self.G.nodes)-1)
        with torch.no_grad():
            diff = self.X[i] - self.X[j]
            norm = diff.norm()
            r = (norm-self.D[i,j]) / 2  / (norm+0.001) * diff
            final_lr = min(0.99, self.lr * self.W[i,j])
            grad_xi = r
            grad_xj = -r

            self.X[i].data -= final_lr * grad_xi
            self.X[j].data -= final_lr * grad_xj
            return (final_lr * grad_xi).norm()

    def update(self, steps=10, ):
        maxDiff = 0
        for i in range(steps):
            diff = self.update_ij()
            if diff > maxDiff:
                maxDiff = diff
        return self.X, maxDiff

