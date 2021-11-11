import torch
from torch import nn


import numpy as np
import time
import utils.poly_point_isect as bo   ##bentley-ottmann sweep line
import networkx as nx

from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import scipy.io as io

def best_scale_stress(pos, D, W):
    n,m = pos.shape[0], pos.shape[1]
    x0 = pos.repeat(1, n).view(-1,m)
    x1 = pos.repeat(n, 1)
    D = D.view(-1)
    W = W.view(-1)
    pdist = nn.PairwiseDistance()(x0, x1)
    s = (W*D*pdist).sum() / (W * pdist**2).sum()
    return s


def best_scale_ideal_edge_length(pos, G, k2i, targetLengths=None):
    if targetLengths is None:
        targetLengths = {e:1 for e in G.edges}
        
    edges = G.edges
    sourceIndices, targetIndices = zip(*[ [k2i[e0], k2i[e1]] for e0,e1 in edges])
    source = pos[sourceIndices,:]
    target = pos[targetIndices,:]
    edgeLengths = (source-target).norm(dim=1)
    targetLengths = torch.tensor([targetLengths[e] for e in edges])

    s = (edgeLengths-targetLengths).sum() / (edgeLengths-targetLengths).pow(2).sum()
    
    return s
    
    
    
def criterion_to_title(criterion):
    return ' '.join([x.capitalize() for x in criterion.split('_')])


def are_edge_pairs_crossed(p):
    '''
    p - positions of n pairs edges in a [n,8] pytorch tensor, 
        where the postions of 8 nodes come in [ax, ay, bx, by, 
        cx, cy, dy, dy] for the edge pair a-b and c-d.
    
    return - an 1D tensor of boolean values, 
             where True means two edges cross each other. 
    '''
    p1, p2, p3, p4 = p[:,:2], p[:,2:4], p[:,4:6], p[:,6:]
    a = p2 - p1
    b = p3 - p4
    c = p1 - p3
    ax, ay = a[:,0], a[:,1]
    bx, by = b[:,0], b[:,1]
    cx, cy = c[:,0], c[:,1]
    
    denom = ay*bx - ax*by
    numer_alpha = by*cx-bx*cy
    numer_beta = ax*cy-ay*cx
    alpha = numer_alpha / denom
    beta = numer_beta / denom
    return torch.logical_and(
        torch.logical_and(0<alpha, alpha<1),
        torch.logical_and(0<beta, beta<1),
    )

def shortest_path(G):
    k2i = {k:i for i,k in enumerate(G.nodes)}
    edge_indices = np.array([(k2i[n0], k2i[n1]) for (n0,n1) in G.edges])
    row_indices = edge_indices[:,0]
    col_indices = edge_indices[:,1]
    adj_data = np.ones(len(edge_indices))
    adj_sparse = csr_matrix((
        adj_data, 
        (row_indices, col_indices)
    ), shape=(len(G), len(G)), dtype=np.float32)

    D = csgraph.shortest_path(adj_sparse, directed=False, unweighted=True)
    return D, adj_sparse, k2i



def load_spx_teaser():
    return load_node_link_txt('./input_graphs/spx_teaser.txt')
    
    
def load_node_link_txt(fn):
    
    def skip_nodes(f,n):
        for _ in range(n):
            f.readline()

    G = nx.Graph()
    with open(fn) as f:
        n = int(f.readline().strip())
        skip_nodes(f, n)
        G.add_nodes_from(range(n))
        for e in f:
            e = e.strip().split()
            e = int(e[0]), int(e[1])
            G.add_edge(*e)
    return G


def load_mat(fn='input_graphs/SuiteSparse Matrix Collection/grid1_dual.mat'):
    
    ## credit:
    ## https://github.com/jxz12/s_gd2/blob/master/jupyter/main.ipynb

    # load the data from the SuiteSparse Matrix Collection format
    # https://www.cise.ufl.edu/research/sparse/matrices/

    mat_data = io.loadmat(fn)
    adj = mat_data['Problem']['A'][0][0]
    G = nx.convert_matrix.from_numpy_matrix(adj.toarray())
    return G




def get_angles(rays):
    x,y = rays[:,0], rays[:,1]
    thetas = torch.angle(x+y*1j)
    thetas_sorted = torch.sort(thetas).values
    angles = thetas_sorted.roll(-1) - thetas_sorted
    angles[-1] *= -1
#     angles[angles>np.pi] = 2*np.pi - angles[angles>np.pi]
    return angles

    
def tick(t0, msg=''):
    t1 = time.time()
    print(f'{msg}: {t1-t0}')
    return t1

def sample_edges(G, sampleSize):
    edge_list = list(G.edges)
    sample_indices = np.random.choice(len(edge_list), min(sampleSize,len(edge_list)), replace=False)
    edge_samples = [edge_list[i] for i in sample_indices]
    return edge_samples

def sample_nodes(G, sampleSize):
    node_list = list(G.nodes)
    sample_indices = np.random.choice(len(node_list), min(sampleSize,len(node_list)), replace=False)
    node_samples = [node_list[i] for i in sample_indices]
    return node_samples

def sample_crossings(pos, G, k2i, sampleSize, sampleOn):
    edge_list = list(G.edges)
    if sampleOn == 'edges':
        sample_indices = np.random.choice(len(edge_list), min(sampleSize,len(edge_list)), replace=False)
        edge_samples = [edge_list[i] for i in sample_indices]
        crossing_segs_sample = find_crossings(pos, edge_samples, k2i)
        
    elif sampleOn == 'crossings':
        crossing_segs = find_crossings(pos, edge_list, k2i)
        crossing_count = crossing_segs.shape[0]
        sample_indices = np.random.choice(crossing_count, min(sampleSize,crossing_count), replace=False)
        crossing_segs_sample = crossing_segs[sample_indices]
    return crossing_segs_sample
    
    

def find_crossings(pos, G_edges, k2i):
    
    ##TODO improve runtime
    t0 = time.time()
    x = pos.detach().cpu().numpy().tolist()
#     t0 = tick(t0, 'a')
    x = [(
            tuple([*x[k2i[e0]], k2i[e0]]), ## (x, y, source id)
            tuple([*x[k2i[e1]], k2i[e1]])  ## (x, y, target id) 
        )
        for e0,e1 in G_edges
    ]
#     t0 = tick(t0, 'b')
 
    ## option 1
    point_segs_pairs = bo.isect_segments_include_segments(x)
    crossing_segs = np.array([psp[1][:2] for psp in point_segs_pairs])##select first [:2] edges whenever more than 2 edges crossed at the same intersection 
    
#     t0 = tick(t0, 'c')
    if len(crossing_segs) > 0:
        crossing_segs = crossing_segs[:,:,:,2].reshape([crossing_segs.shape[0], -1])
        crossing_segs = crossing_segs.astype(np.int)## indices of 4 nodes in edge crossing pairs
        return crossing_segs
    else:
        return np.zeros([0,4])

#     ## option 2 (faster)
#     crossing_segs = []
#     intersections = intersection(x)
#     if intersections is not None:
#         for l in list(intersections.values()):
#             a,b,c,d = l[0][0][2], l[0][1][2], l[1][0][2], l[1][1][2]
#             if a!=c and a!=d and b!=c and b!=d:
#                 crossing_segs.append([a,b,c,d])
#         crossing_segs = np.array(crossing_segs)
#         return crossing_segs
#     #     t0 = tick(t0, 'c')
#     else:
#         return np.zeros([0,4])

    
def count_crossings(pos, edge_pair_indices):

    x = pos.detach().cpu().numpy().tolist()
#     t0 = tick(t0, 'a')
    x = [(
            tuple([*x[e0]]), ## (x, y)
            tuple([*x[e1]]) 
        )
        for e0,e1 in edge_pair_indices
    ]
    return len(bo.isect_segments(x))


#https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
def pairwise_distance_squared(x, y=None, w=None):
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
    return dist


def file2graph(fn='./facebook/0.edges'):
    with open(fn) as f:
        lines = [l.split()[:2] for l in f.readlines()]
        edges = [tuple(int(i) for i in l) for l in lines]
        nodes = set(sum(edges, ())) ## SLOW?
#         edges += [(-1, n) for n in nodes]
#         nodes.update({-1})
    G = nx.Graph()
    G.add_nodes_from(list(nodes))
    G.add_edges_from(edges)
    return G



def dict2tensor(d, device='cpu', fill=None):
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