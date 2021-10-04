from utils import utils
from utils import vis as V

import criteria as C
import quality as Q

import time
from tqdm.notebook import tqdm

import numpy as np

import torch
from torch import nn, optim

##TODO custom lr_scheduler, linear slowdown on plateau 

class GD2:
    def __init__(self, G, device='cpu'):
        self.G = G
        
        
        self.D, self.adj_sparse, self.k2i = utils.shortest_path(G)
        
        self.adj = torch.from_numpy((self.adj_sparse+self.adj_sparse.T).toarray())
        self.D = torch.from_numpy(self.D)
        self.i2k = {i:k for k,i in self.k2i.items()}
        self.W = 1/(self.D**2+1e-6)

        self.degrees = np.array([self.G.degree(self.i2k[i]) for i in range(len(self.G))])
        self.maxDegree = max(dict(self.G.degree).values())

        self.edge_indices = [(self.k2i[e0], self.k2i[e1]) for e0,e1 in self.G.edges]
        self.node_indices = range(len(self.G))
        self.node_index_pairs = np.c_[
            np.repeat(self.node_indices, len(self.G)),
            np.tile(self.node_indices, len(self.G))
        ]
        self.stress_sample_start = 0
        np.random.shuffle(self.node_index_pairs)
        
        ## init
        self.pos = torch.randn(len(self.G.nodes), 2, device=device).requires_grad_(True)
        self.qualities_by_time = []
        self.i = 0
        self.runtime = 0
        self.loss_curve = []
        

    def optimize(self,
        criteria_weights={'stress':1.0}, 
        sample_sizes={'stress':128},
        evaluate=None,
        evaluate_interval=None,
        max_iter=int(1e4),
        grad_clamp=4,
        vis=False,
        vis_interval=100,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
    ):
        
        ## shortcut of object attributes
        G = self.G
        D, adj_sparse, k2i = self.D, self.adj_sparse, self.k2i
        W = self.W
        pos = self.pos
        
        ## measure runtime
        t0 = time.time()

        ## prepare training
        optimizer_kwargs_default = dict(
            lr=1, momentum=0.7, nesterov=True
        )
        if optimizer_kwargs is not None:
            for k,v in optimizer_kwargs.items():
                optimizer_kwargs_default[k] = v
        optimizer_kwargs = optimizer_kwargs_default
        optimizer = optim.SGD([pos], **optimizer_kwargs)
        
#         patience = np.ceil(np.log2(len(G)+1))*100
#         if 'stress' in criteria_weights and sample_sizes['stress'] < 16:
#             patience += 100 * 16/sample_sizes['stress']
        patience = np.ceil(np.log2(len(G)+1))*300
    
        scheduler_kwargs_default = dict(
            factor=0.9, 
            patience=patience, 
            min_lr=1e-5, 
            verbose=True
        )
        if scheduler_kwargs is not None:
            for k,v in scheduler_kwargs.items():
                scheduler_kwargs_default[k] = v
        scheduler_kwargs = scheduler_kwargs_default
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

        ## start training
        iterBar = tqdm(range(max_iter))
        stopCountdown = 100

        for _ in iterBar:
            t0 = time.time()
            ## optimization
            optimizer.zero_grad()

            loss = 0
            for c, weight in criteria_weights.items():
                if weight == 0:
                    continue
                    
                if c == 'stress':
                    if self.stress_sample_start >= len(self.node_index_pairs):
                        np.random.shuffle(self.node_index_pairs)
                        self.stress_sample_start = 0

                    stress_samples = self.node_index_pairs[
                        self.stress_sample_start : self.stress_sample_start+sample_sizes['stress']
                    ]
                    self.stress_sample_start+=sample_sizes['stress']

                    loss += weight * C.stress(
                        pos, D, W, 
                        samples=stress_samples, reduce='mean',)


                elif c == 'edge_uniformity':
                    loss += weight * C.edge_uniformity(
                        pos, G, k2i, sample_sizes['edge_uniformity']-1)


                elif c == 'neighborhood_preservation':
                    loss += weight * C.neighborhood_preseration(
                        pos, G, adj, 
                        k2i, i2k, 
                        degrees, maxDegree,
                        n_roots=sample_sizes['neighborhood_preservation'], 
                        depth_limit=2)


                elif c == 'crossings':
                    loss += weight * C.crossings(
                        pos, G, k2i, reg_coef=0.01, 
                        niter=20, 
                        sampleSize=sample_sizes['crossings'], sampleOn='crossings'
                    )


                elif c == 'crossing_angle_maximization':
                    loss += weight * C.crossing_angle_maximization(
                        pos, G, k2i, i2k, 
                        sampleSize=sample_sizes['crossing_angle_maximization'], 
                        sampleOn='crossings') ## slow for large sample size


                elif c == 'aspect_ratio':
                    loss += weight * C.aspect_ratio(
                        pos, sampleSize=sample_sizes['crossing_angle_maximization'])


                elif c == 'angular_resolution':
#                     loss += weight * C.angular_resolution(pos, G, k2i, sampleSize=sampleSize//maxDegree)
                    loss += weight * C.angular_resolution(
                        pos, G, k2i, sampleSize=sample_sizes['angular_resolution'])


                elif c == 'vertex_resolution':
                    loss += weight * C.vertex_resolution(
                        pos, sampleSize=sample_sizes['vertex_resolution'], target=1/len(G)**0.5)


                elif c == 'gabriel':
#                     loss += weight * C.gabriel(pos, G, k2i, sampleSize=int(sampleSize**0.5))
                    loss += weight * C.gabriel(
                      pos, G, k2i, sampleSize=sample_sizes['gabriel'])


                else:
                    print(f'Criteria not supported: {c}')

                
            loss.backward()
            pos.grad.clamp_(-grad_clamp, grad_clamp)
            optimizer.step()

            
            self.runtime += time.time() - t0

            if vis and self.i%vis_interval==vis_interval-1:
                pos_numpy = pos.detach().cpu().numpy()
                pos_G = {k:pos_numpy[k2i[k]] for k in G.nodes}
                V.plot(
                    G, pos_G,
                    self.loss_curve, 
                    self.i, self.runtime, 
                    edge=True, show=True, save=False
                )



            if loss.isnan():
                raise Exception('loss is nan')
                break
            if pos.isnan().any():
                raise Exception('pos is nan')
                break

            if self.i % 100 == 99:
                iterBar.set_postfix({'loss': loss.item(), })    

            if len(self.loss_curve) > 0:
                self.loss_curve.append(0.999*self.loss_curve[-1] + 0.001*loss.item())
            else:
                self.loss_curve.append(loss.item())

            if scheduler is not None:
                scheduler.step(self.loss_curve[-1])


            lr = optimizer.param_groups[0]['lr']

            # maxGrad = pos.grad.norm(dim=1).max()
    #         maxGrad = pos.grad.abs().max()
    #         minStepSize = 1e-6
    #         if lr * maxGrad <= minStepSize:
    #             stopCountdown -= 1
    #             if stopCountdown <= 0:
    #                 break

            if lr <= scheduler.min_lrs[0]:
                break

            if evaluate is not None and evaluate_interval and self.i%evaluate_interval==evaluate_interval-1:
                qualities = self.evaluate(qualities=evaluate)
                self.qualities_by_time.append(dict(
                    time=self.runtime,
                    iter=self.i,
                    qualities=qualities
                ))
                
            self.i+=1
            
                
        
        ## attach pos to G.nodes        
        pos_numpy = pos.detach().numpy()
        for k in G.nodes:
            G.nodes[k]['pos'] = pos_numpy[k2i[k],:]
        
        ## prepare result
        result = self.get_result_dict(evaluate, sample_sizes)
        return result

    
    def get_result_dict(self, evaluate, sample_sizes):
        return dict(
            iter=self.i, 
            loss_curve=self.loss_curve,
            runtime=self.runtime,
            qualities_by_time=self.qualities_by_time,
            qualities=self.evaluate(qualities=evaluate) if evaluate is not None else None,
            sample_sizes=sample_sizes
        )
    
    
    def evaluate(
        self,
        pos=None,
        qualities={'stress'},
        verbose=False
    ):
        
        if pos is None:
            pos = self.pos
            
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
            if verbose:
                print(f'Evaluating {q}...', end='')
                
            t0 = time.time()
            if q == 'stress':
                qualityMeasures[q] = Q.stress(pos, self.D, self.W, None)
            elif q == 'edge_uniformity':
                qualityMeasures[q] = Q.edge_uniformity(pos, self.G, self.k2i)
            elif q == 'neighborhood_preservation':
                qualityMeasures[q] = Q.neighborhood_preservation(pos, self.G, self.adj, self.i2k)
            elif q == 'crossings':
                qualityMeasures[q] = Q.crossings(pos, self.edge_indices)
            elif q == 'crossing_angle_maximization':
                qualityMeasures[q] = Q.crossing_angle_maximization(pos, self.G.edges, self.k2i)
            elif q == 'aspect_ratio':
                qualityMeasures[q] = Q.aspect_ratio(pos)
            elif q == 'angular_resolution':
                qualityMeasures[q] = Q.angular_resolution(pos, self.G, self.k2i)
            elif q == 'vertex_resolution':
                qualityMeasures[q] = Q.vertex_resolution(pos, target=1/len(self.G)**0.5)
            elif q == 'gabriel':
                qualityMeasures[q] = Q.gabriel(pos, self.G, self.k2i)
           
            if verbose:
                print(f'done in {time.time()-t0:.2f}s')
        return qualityMeasures