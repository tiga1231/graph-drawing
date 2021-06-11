import torch
import numpy as np
import time
import poly_point_isect as bo   ##bentley-ottmann sweep line



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