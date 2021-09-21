from utils import utils
import criteria as C
import quality as Q

import time
from tqdm.notebook import tqdm

import numpy as np

import torch
from torch import nn, optim

def optimize(
    G, 
    criteria_weights={'stress':1.0}, 
    sample_sizes={'stress':128},
    evaluate=None, ## TODO
    
    optimizer=None,
    scheduler=None,
    niter=int(1e4),
    gClamp=4,
    device='cpu',
    learning_rate=1,
):
    
    t0 = time.time()
    
    
    D, adj_sparse, k2i = utils.shortest_path(G)
    
    maxDegree = max(dict(G.degree).values())
    adj = torch.from_numpy(adj_sparse.toarray())
    D = torch.from_numpy(D)
    i2k = {i:k for k,i in k2i.items()}
    W = 1/(D**2+1e-6)
    
    edge_indices = [(k2i[e0], k2i[e1]) for e0,e1 in G.edges]
    node_indices = range(len(G))
    node_index_pairs = np.c_[
        np.repeat(node_indices, len(G)),
        np.tile(node_indices, len(G))
    ]
    stress_sample_start = 0
    np.random.shuffle(node_index_pairs)
    

    ##training
    pos = torch.randn(len(G.nodes), 2, device=device).requires_grad_(True)

    if optimizer is None:
        optimizer = optim.SGD([pos], lr=learning_rate, momentum=0.7, nesterov=True)
    else:
        optimizer = optimizer([pos])
    if scheduler is None:
        patience = np.ceil(np.log2(len(G)/sample_sizes['stress']+1))*100
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, 
            patience=patience, 
            min_lr=1e-5, verbose=False
        )
    else:
        scheduler = scheduler(optimizer)


    iterBar = tqdm(range(niter))
    lossCurve = []
    stopCountdown = 100
    
    for i in iterBar:
        ## optimization
        optimizer.zero_grad()

        loss = 0
        for c, weight in criteria_weights.items():
            if weight != 0:
                if c == 'stress':
                    if stress_sample_start >= len(node_index_pairs):
                        np.random.shuffle(node_index_pairs)
                        stress_sample_start = 0
                    stress_samples = node_index_pairs[
                        stress_sample_start : stress_sample_start+sample_sizes['stress']
                    ]
                    stress_sample_start+=sample_sizes['stress']

                    loss += weight * C.stress(
                        pos, D, W, samples=stress_samples
                    )
                elif c == 'edge_uniformity':
                    loss += weight * C.edge_uniformity(
                        pos, G, k2i, sample_sizes['edge_uniformity']-1
                    )
                elif c == 'neighborhood_preseration':
                    loss += weight * C.neighborhood_preseration(
                        pos, G, adj, k2i, i2k, n_roots=2, depth_limit=2
                    )
                elif c == 'crossings':
                    loss += weight * C.crossings(
                        pos, G, k2i, reg_coef=0.01, niter=20, sampleSize=sample_sizes['crossings'], sampleOn='crossings'
                    )
                elif c == 'crossing_angle_maximization':
                    loss += weight * C.crossing_angle_maximization(
                        pos, G, k2i, i2k, sampleSize=sample_sizes['crossing_angle_maximization'], sampleOn='crossings'
                    ) ## slow for large sample size
                elif c == 'aspect_ratio':
                    loss += weight * C.aspect_ratio(
                        pos, sampleSize=sample_sizes['crossing_angle_maximization']
                    )
                elif c == 'angular_resolution':
#                     loss += weight * C.angular_resolution(pos, G, k2i, sampleSize=sampleSize//maxDegree)
                    loss += weight * C.angular_resolution(
                        pos, G, k2i, sampleSize=sample_sizes['angular_resolution']
                    )
                elif c == 'vertex_resolution':
                    loss += weight * C.vertex_resolution(
                        pos, sampleSize=sample_sizes['vertex_resolution'], target=1/len(G)**0.5
                    )
                elif c == 'gabriel':
#                     loss += weight * C.gabriel(pos, G, k2i, sampleSize=int(sampleSize**0.5))
                    loss += weight * C.gabriel(
                      pos, G, k2i, sampleSize=sample_sizes['gabriel']
                    )

                    
                else:
                    print(f'Criteria not supported: {c}')

        loss.backward()
        pos.grad.clamp_(-gClamp, gClamp)
        optimizer.step()

        
        if loss.isnan():
            raise Exception('loss is nan')
            break
        if pos.isnan().any():
            raise Exception('pos is nan')
            break
        
        if i % 100 == 99:
            iterBar.set_postfix({'loss': loss.item(), })    
        
        if len(lossCurve) > 0:
            lossCurve.append(0.9*lossCurve[-1] + 0.1*loss.item())
        else:
            lossCurve.append(loss.item())

        if scheduler is not None:
            scheduler.step(lossCurve[-1])

        lr = optimizer.param_groups[0]['lr']
        # maxGrad = pos.grad.norm(dim=1).max()
        maxGrad = pos.grad.abs().max()
        minStepSize = 1e-3
        if lr * maxGrad <= minStepSize:
            stopCountdown -= 1
            if stopCountdown <= 0:
                break

    pos_numpy = pos.detach().numpy()
    for k in G.nodes:
        G.nodes[k]['pos'] = pos_numpy[k2i[k],:]
    
    totalTime = time.time() - t0

    result = dict(
        G=G,
        pos=pos, 
        i2k=i2k, ##todo remove this and store in G.node data
        k2i=k2i, ##todo remove this and store in G.node data
        iter=i, 
        loss_curve=lossCurve,
        runtime=totalTime,
        shortest_path_distance=D,
        stress_weight_matrix=W,
        adjacency_matrix=adj_sparse,
        edge_indices=edge_indices,
    )
    
    if evaluate is not None:
        result['qualities'] = _evaluate(result, qualities=evaluate)
        
    return result


def _evaluate(
    result,
    qualities={'stress'},
):
    pos = result['pos']
    G = result['G']
    k2i = result['k2i']
    i2k = result['i2k']
    D = result['shortest_path_distance']
    W = result['stress_weight_matrix']
    edge_indices = result['edge_indices']
    adj = torch.from_numpy(result['adjacency_matrix'].toarray())

    qualityMeasures = dict()
    if qualities == 'all':
        qualities = {
            'stress',
            'edge_uniformity',
            'neighborhood_preservation',
            'crossings',
            'crossing_angle_maximization',
            'aspect_ratio',
            'angular_resolution',
            'vertex_resolution',
            'gabriel',
        }
        
        
    for q in qualities:
        print(f'Evaluating {q}...', end='')
        t0 = time.time()
        if q == 'stress':
            qualityMeasures[q] = Q.stress(pos, D, W, None)
        elif q == 'edge_uniformity':
            qualityMeasures[q] = Q.edge_uniformity(pos, G, k2i)
        elif q == 'neighborhood_preservation':
            qualityMeasures[q] = Q.neighborhood_preservation(pos, G, adj, i2k)
        elif q == 'crossings':
            qualityMeasures[q] = Q.crossings(pos, edge_indices)
        elif q == 'crossing_angle_maximization':
            qualityMeasures[q] = Q.crossing_angle_maximization(pos, G.edges, k2i)
        elif q == 'aspect_ratio':
            qualityMeasures[q] = Q.aspect_ratio(pos)
        elif q == 'angular_resolution':
            qualityMeasures[q] = Q.angular_resolution(pos, G, k2i)
        elif q == 'vertex_resolution':
            qualityMeasures[q] = Q.vertex_resolution(pos, target=1/len(G)**0.5)
        elif q == 'gabriel':
            qualityMeasures[q] = Q.gabriel(pos, G, k2i)
        print(f'done in {time.time()-t0:.2f}s')
    return qualityMeasures