# from pynndescent import NNDescent
from utils import lovasz_losses as L
from utils import utils

import torch
from torch import nn
from torch import optim
import numpy as np
import networkx as nx

import random

    
    
    
# def crossings(pos, G, k2i, sampleSize, sampleOn='edges', reg_coef=1, niter=30):
#     crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn)
# #     if len(crossing_segs_sample) < sampleSize*0.5:
#     if sampleOn=='edges' and len(crossing_segs_sample) == 0:
#         crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn='crossings')
        
#     if len(crossing_segs_sample) > 0:
#         pos_segs = pos[crossing_segs_sample.flatten()].view(-1,4,2)
#         w = (torch.rand(pos_segs.shape[0], 2, 1)-0.5).requires_grad_(True)
#         b = (torch.rand(pos_segs.shape[0], 1, 1)-0.5).requires_grad_(True)
#         relu = nn.ReLU()
#         o = optim.SGD([w,b], lr=0.01, momentum=0.5, nesterov=True)
#         for _ in range(niter):
#             pred = pos_segs.detach() @ w + b
#             ## assume labels of nodes in the first edges are -1
#             ## now flip the pred of those nodes so that now we want every pred to be +1
#             pred[:,:2,:] = -pred[:,:2,:]
            
#             loss_svm = relu(1-pred).sum() + reg_coef * w.pow(2).sum()
#             o.zero_grad()
#             loss_svm.backward()
#             o.step()
#         pred = pos_segs @ w.detach() + b.detach()
    
#         pred[:,:2,:] = -pred[:,:2,:] 
#         loss_crossing = relu(1-pred).sum()
#         return loss_crossing
#     else:
#         ##return dummy loss
#         return (pos[0,0]*0).sum()
    
    



# def angular_resolution(pos, G, k2i, sampleSize=2):
#     samples = utils.sample_nodes(G, sampleSize)
#     neighbors = [list(G.neighbors(s)) for s in samples]
#     sampleIndices = [k2i[s] for s in samples]
#     neighborIndices = [[k2i[n] for n in nei] for nei in neighbors]
    
#     samples = pos[sampleIndices]
#     neighbors = [pos[nei] for nei in neighborIndices]
    
#     angles = [utils.get_angles(rs) for rs in rays]
#     if len(angles) > 0:
#         loss = sum([torch.exp(-a*len(a)).sum() for a in angles])
# #         loss = sum([(a - np.pi*20/len(a)).pow(2).sum() for a in angles])
#     else:
#         loss = pos[0,0]*0##dummy output
#     return loss

def angular_resolution(pos, G, k2i, sampleSize=2, sample=None):
    relu = nn.ReLU()
    
    
    if sample is None:
        sample = utils.sample_nodes(G, sampleSize)
        sample = [s for s in sample if len(list(G.neighbors(s)))>=2]

        if len(sample)==0:
            loss = pos[0,0]*0##dummy output
            return loss
        degrees = torch.tensor([G.degree(s) for s in sample])
        neighbors = [random.choices(list(G.neighbors(s)),k=2) for s in sample]
        sampleIndices = [k2i[s] for s in sample]
        neighborIndices = [[k2i[n] for n in nei] for nei in neighbors]
        sample = pos[sampleIndices]
        neighbors = torch.stack([pos[nei] for nei in neighborIndices])
        rays = neighbors - sample.unsqueeze(1)
        sim = cos_sim(rays[:,0,:], rays[:,1,:])
        
    else:
        degrees, a,b,c,d = sample
        a,b,c,d = pos[a], pos[b], pos[c], pos[d]
        sim = cos_sim(b-a, d-c)
        
    angles = torch.acos(sim.clamp_(-0.99,0.99))
    if sim.shape[0] > 0:
#         loss = ((cos+1)**2 / 4).sum()
#         loss = torch.exp(-angles).sum()
#         loss = relu(-angles + 2*np.pi/degrees).pow(2).mean()

        optimal = 2*np.pi/degrees
        loss = bce(relu((-angles + optimal)/optimal), torch.zeros_like(angles))
#         loss = torch.exp(-angles/optimal).mean()
    else:
        loss = pos[0,0]*0##dummy output
    return loss



def gabriel(pos, G, k2i, sample=None, sampleSize=64):
    
    if sample is not None:
        nodes, edges = sample
        edges = torch.stack(edges, dim=-1)
    else:
        edges = utils.sample_edges(G, sampleSize)
        nodes = utils.sample_nodes(G, sampleSize)
    
        edges = np.array([(k2i[e0], k2i[e1]) for e0,e1 in edges])
        nodes = np.array([k2i[n] for n in nodes])
    m,n = len(nodes), len(edges)
    node_pos = pos[nodes]
    edge_pos = pos[edges.flatten()].reshape([-1,2,2])
    centers = edge_pos.mean(1)
    radii = (edge_pos[:,0,:] - edge_pos[:,1,:]).norm(dim=1, keepdim=True)/2
    
#     centers = centers.repeat(1,m).view(-1, 2)
#     radii = radii.repeat(1,m).view(-1, 1)
#     node_pos = node_pos.repeat(n,1)
    
    relu = nn.ReLU()
#     print((node_pos-centers).norm(dim=1))
    loss = relu(radii+0.01 - (node_pos-centers).norm(dim=1)).pow(2)
#     loss = relu(radii - (node_pos-centers).norm(dim=1)).pow(2)
    loss = loss.mean()
    return loss


bce = torch.nn.BCELoss()
cos_sim = torch.nn.CosineSimilarity()
def crossing_angle_maximization(pos, G, k2i, i2k, sample=None, sample_labels=None, sampleSize=16, sampleOn='edges'):
#     edge_list = list(G.edges)
#     if sampleOn == 'edges':
#         sample_indices = np.random.choice(len(edge_list), sampleSize, replace=False)
#         edge_samples = [edge_list[i] for i in sample_indices]
#         crossing_segs_sample = utils.find_crossings(pos, edge_samples, k2i)
        
#     elif sampleOn == 'crossings':
#         crossing_segs = utils.find_crossings(pos, edge_list, k2i)
#         crossing_count = crossing_segs.shape[0]
#         sample_indices = np.random.choice(crossing_count, min(sampleSize,crossing_count), replace=False)
#         crossing_segs_sample = crossing_segs[sample_indices]

#     if len(crossing_segs_sample) > 0:
#     pos_segs = pos[crossing_segs_sample.flatten()].view(-1,4,2)


    if sample is None:
        crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn)
        if sampleOn=='edges' and len(crossing_segs_sample) == 0:
            crossing_segs_sample = utils.sample_crossings(pos, G, k2i, sampleSize, sampleOn='crossings')
            sample = crossing_segs_sample
            
    pos_segs = pos[sample.flatten()].view(-1,4,2)
    if sample_labels is None:
        sample_labels = utils.are_edge_pairs_crossed(pos_segs.view(-1,8))
    
    
            
    pos_segs = pos[sample.flatten()].view(-1,4,2)
    v1 = pos_segs[:,1] - pos_segs[:,0]
    v2 = pos_segs[:,3] - pos_segs[:,2]
    sim = cos_sim(v1, v2)
    
#     angles = torch.acos(sim.clamp_(-0.99,0.99))
#     return (sample_labels*(angles-np.pi/2)).pow(2).mean()
    
#     return (sample_labels * sim**2).mean()
    return (sample_labels * sim**2 / (1-sim**2+1e-6)).mean()
#     return bce(
#         sample_labels * (sim**2).clamp_(1e-4, 1-1e-4), 
#         torch.zeros_like(sim)
#     )
#     else:
#         return pos[0,0]*0##dummy loss
    
    
def aspect_ratio(pos, sampleSize=None, sample=None,
#                  rotation_angles=torch.arange(7,dtype=torch.float)/7*(np.pi/2), 
                 target=[1,1]):
    
    if target[0] < target[1]:
        target = target[::-1]
        
    if sample is not None:
        sample = pos[sample,:]
    elif sampleSize is None or sampleSize=='full':
        sample = pos
    else:
        n = pos.shape[0]
        i = np.random.choice(n, min(n,sampleSize), replace=False)
        sample = pos[i,:]
    bce = nn.BCELoss(reduction='sum')
    singular_values = torch.svd(sample-sample.mean(dim=0)).S
    
    return bce(singular_values[1]/singular_values[0], torch.tensor(target[1]/target[0]))

        
#     mean = sample.mean(dim=0, keepdim=True)
#     sample -= mean
#     scale = sample.max().detach()
#     sample = sample/scale * 1
    
#     cos = torch.cos(rotation_angles)
#     sin = torch.sin(rotation_angles)
#     rot = torch.stack([cos, sin, -sin, cos], 1).view(len(rotation_angles), 2, 2)
    
#     sample = sample.matmul(rot)

#     softmax = nn.Softmax(dim=1)
# #     print(softmax(sample))
#     max_hat = (softmax(sample) * sample).sum(1)
#     min_hat = (softmax(-sample) * sample).sum(1)
    
#     w = max_hat[:,0] - min_hat[:,0]
#     h = max_hat[:,1] - min_hat[:,1]
#     estimate = torch.stack([w,h], 1)
#     estimate /= estimate.sum(1, keepdim=True)
# #     print(estimate)
#     target = torch.tensor(target_width_to_height, dtype=torch.float)
#     target /= target.sum()
#     target = target.repeat(len(rotation_angles), 1)
#     bce = nn.BCELoss(reduction='mean')
#     mse = nn.MSELoss(reduction='mean')
#     return mse(estimate, target)



def vertex_resolution(pos, sampleSize=None, sample=None, target=0.1, prev_target_dist=1, prev_weight=1):
    pairwiseDistance = nn.PairwiseDistance()
    relu = nn.ReLU()
#     softmax = nn.Softmax(dim=0)
#     softmin = nn.Softmin(dim=0)
    
    n = pos.shape[0]
    if sample is None:
        if sampleSize is None or sampleSize=='full':
            sample = pos
        else:
            i = np.random.choice(n, min(n,sampleSize), replace=False)
            sample = pos[i,:]
        m = sample.shape[0]
        a = sample.repeat([1,m]).view(-1,2)
        b = sample.repeat([m,1])
        pdist = pairwiseDistance(a, b)
    
    else:
        a = pos[sample[:,0],:]
        b = pos[sample[:,1],:]
        pdist = pairwiseDistance(a,b)
    
#     dmax = (softmax(pdist)*pdist).sum().detach()
    dmax = pdist.max().detach()
    
    target_dist = target*dmax
    ## exponentially smoothed target_dist
    smoothness = 0.1
    weight = prev_weight*smoothness + 1
    target_dist = (
        max(target_dist, prev_target_dist) 
        +min(target_dist, prev_target_dist)*smoothness
    )/weight
    
#     loss = len(pdist)*(softmin(pdist).detach() * relu((target_dist - pdist)/target_dist)).sum()
#     loss = relu((target_dist - pdist)/targetDist).sum()
#     loss = relu(target_dist - pdist).sum()
    loss = relu(1 - pdist/target_dist).pow(2).mean()
    return loss, target_dist, weight


def neighborhood_preseration(pos, G, adj, k2i, i2k, 
                             degrees, max_degree,
                             sample=None,
                             n_roots=2, 
                             depth_limit=1, 
                             neg_sample_rate=0.5, 
                             device='cpu'):
    
    if sample is not None:
        pos_samples = []
        for root_index in sample:
            root = i2k[root_index]
            G_sub = nx.bfs_tree(G, root, depth_limit=depth_limit)
            pos_samples += [k2i[n] for n in G_sub.nodes]
        pos_samples = sorted(set(pos_samples))
    else:
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
    
    ## pdist
    x0 = x.repeat(1, n).view(-1,m)
    x1 = x.repeat(n, 1)
    pdist = nn.PairwiseDistance()(x0, x1).view(n, n)
    
    
    ## k_dist
    # Option 1, mean of k-th and (k+1)th-NN as threshold
    # degrees = degrees[samples]
    # max_degree = degrees.max()
    # n_neighbors = max(2, min(max_degree+1, n))
    # n_trees = min(64, 5 + int(round((n) ** 0.5 / 20.0)))
    # n_iters = max(5, int(round(np.log2(n))))
    # knn_search_index = NNDescent(
    #     x.detach().numpy(),
    #     n_neighbors=n_neighbors,
    #     n_trees=n_trees,
    #     n_iters=n_iters,
    #     max_candidates=60,
    # )
    # knn_indices, knn_dists = knn_search_index.neighbor_graph
    # kmax = knn_dists.shape[1]-1
    # k_dist = np.array([
    #     ( knn_dists[i,min(kmax, k)] + knn_dists[i,min(kmax, k+1)] ) / 2 
    #     for i,k in enumerate(degrees)
    # ])
    # pred = torch.from_numpy(k_dist.astype(np.float32)).view(-1,1) - pdist

    
    ## Option 2, fraction of diameter of drawing as threshold
    ## TODO adaptive dist (e.g. normalize by diameter of the drawing, 0.1*diameter)
#     diameter = pdist.max().item()
#     k_dist = 0.1 * max(diameter, 1.0)
#     pred = -pdist + k_dist

    ## Option 3, constant threshold
    k_dist = 1.5
    pred = -pdist + k_dist
    # pred = torch.sign(-pdist + k_dist) * (-pdist + k_dist).pow(2)
    
    ## Option 4 (TODO), threshold by furthest neighboring node
    
    

    
    target = adj + torch.eye(adj.shape[0], device=device)
    loss = L.lovasz_hinge(pred, target)
    return loss




def ideal_edge_length(pos, G, k2i, targetLengths=None, sampleSize=None, sample=None, reduce='mean'):
    if targetLengths is None:
        targetLengths = {e:1 for e in G.edges}
        
    n,m = pos.shape[0], pos.shape[1]
    if sample is not None:
        edges = sample
    elif sampleSize is not None:
        edges = random.sample(G.edges, sampleSize)
    else:
        edges = G.edges

    sourceIndices, targetIndices = zip(*[ [k2i[e0], k2i[e1]] for e0,e1 in edges])
    source = pos[sourceIndices,:]
    target = pos[targetIndices,:]
    edgeLengths = (source-target).norm(dim=1)
    targetLengths = torch.tensor([targetLengths[e] for e in edges])

    eu = ((edgeLengths-targetLengths)/targetLengths).pow(2)
    
    if reduce == 'sum':
        return eu.sum()
    elif reduce == 'mean':
        return eu.mean()




def stress(pos, D, W, sampleSize=None, sample=None, reduce='mean'):
    if sample is None:
        n,m = pos.shape[0], pos.shape[1]
        if sampleSize is not None:
            i0 = np.random.choice(n, sampleSize)
            i1 = np.random.choice(n, sampleSize)
            x0 = pos[i0,:]
            x1 = pos[i1,:]
            
        
            D = torch.tensor([D[i,j] for i, j in zip(i0, i1)])
            W = torch.tensor([W[i,j] for i, j in zip(i0, i1)])
        else:
            x0 = pos.repeat(1, n).view(-1,m)
            x1 = pos.repeat(n, 1)
            D = D.view(-1)
            W = W.view(-1)
    else:
        x0 = pos[sample[:,0],:]
        x1 = pos[sample[:,1],:]
        D = torch.tensor([D[i,j] for i, j in sample])
        W = torch.tensor([W[i,j] for i, j in sample])
    pdist = nn.PairwiseDistance()(x0, x1)
#     wbound = (1/4 * diff.abs().min()).item()
#     W.clamp_(0, wbound)
    
    res = W*(pdist-D)**2
    
    if reduce == 'sum':
        return res.sum()
    elif reduce == 'mean':
        return res.mean()




