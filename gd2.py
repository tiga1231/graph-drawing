from utils import utils
from utils import vis as V
from utils.CrossingDetector import CrossingDetector


import criteria as C
import quality as Q

import time
import itertools

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import pickle as pkl


        
        
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if is_interactive():
    from tqdm.notebook import tqdm
    from IPython import display
else:
    from tqdm import tqdm
    display = None



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
        self.node_index_pairs = self.node_index_pairs[self.node_index_pairs[:,0]<self.node_index_pairs[:,1]]
#         self.stress_sample_start = 0
#         np.random.shuffle(self.node_index_pairs)
        
#         init
        self.pos = torch.randn(len(self.G.nodes), 2, device=device).requires_grad_(True)
#         self.pos = torch.rand(len(self.G.nodes), 2, device=device).requires_grad_(True)
        self.qualities_by_time = []
        self.i = 0
        self.runtime = 0
        self.loss_curve = []
        self.sample_sizes = {}
        
        self.crossing_detector = CrossingDetector()
        self.crossing_detector_loss_fn = nn.BCELoss()
        self.crossing_pos_loss_fn = nn.BCELoss(reduction='sum')
        self.crossing_detector_optimizer = optim.SGD(self.crossing_detector.parameters(), lr=0.1)
        ## filter out incident edge pairs
        self.non_incident_edge_pairs = [
            [self.k2i[e1[0]], self.k2i[e1[1]], self.k2i[e2[0]], self.k2i[e2[1]]] 
            for e1,e2 in itertools.product(G.edges, G.edges) 
            if e1<e2 and len(set(e1+e2))==4
        ]
        self.device='cpu'
        
        
    def grad_clamp(self, l, c, weight, optimizer, ref=1):
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        grad = self.pos.grad.clone()
        grad_norm = grad.norm(dim=1)
        is_large = grad_norm > weight*ref
        grad[is_large] = grad[is_large] / grad_norm[is_large].view(-1,1) * weight*ref
        self.grads[c] = grad
                    
        
    def optimize(self,
        criteria_weights={'stress':1.0}, 
        sample_sizes={'stress':128},
        evaluate=None,
        evaluate_interval=None,
        max_iter=int(1e4),
        grad_clamp=4,
        vis_interval=100,
        clear_output=False,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
                 
    ):

        self.sample_sizes = sample_sizes
        ## shortcut of object attributes
        G = self.G
        D, k2i = self.D, self.k2i
        i2k = self.i2k
        adj = self.adj
        W = self.W
        pos = self.pos
        degrees = self.degrees
        maxDegree = self.maxDegree
        device = self.device
        
        self.init_sampler(criteria_weights)
            
        ## measure runtime
        t0 = time.time()

        ## prepare training
        optimizer_kwargs_default = dict(
            lr=1, 
            momentum=0.7, 
            nesterov=True
        )
        if optimizer_kwargs is not None:
            for k,v in optimizer_kwargs.items():
                optimizer_kwargs_default[k] = v
        optimizer_kwargs = optimizer_kwargs_default
        optimizer = optim.SGD([pos], **optimizer_kwargs)
#         optimizer = optim.RMSprop([pos], **optimizer_kwargs)
        
#         patience = np.ceil(np.log2(len(G)+1))*100
#         if 'stress' in criteria_weights and sample_sizes['stress'] < 16:
#             patience += 100 * 16/sample_sizes['stress']
        patience = np.ceil(np.log2(len(G)+1)) * 300 * (16/min(sample_sizes.values()))
        print(patience)
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

        iterBar = tqdm(range(max_iter))
        stopCountdown = 100

        ## smoothed loss curve during training
        s = 0.5**(1/100) ## smoothing factor for loss curve, setting 'half-life'=100
        weighted_sum_of_loss, sum_of_weight = 0, 0
        vr_target_dist, vr_target_weight = 1, 0
            
        ## start training
        for _ in iterBar:
            t0 = time.time()
            ## optimization
            loss = 0
            self.grads = {}
            ref = 1
            for c, weight in criteria_weights.items():
                if weight == 0:
                    continue
                
                if c == 'stress':
                    sample = self.sample(c)
                    l = weight * C.stress(
                        pos, D, W, 
                        sample=sample, reduce='mean')
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)

                    
                    
                elif c == 'edge_uniformity':
                    l = weight * C.edge_uniformity(
                        pos, G, k2i, targetLengths=None, 
                        sampleSize=sample_sizes[c],
                        reduce='mean',
                    )
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)

                    

                elif c == 'neighborhood_preservation':
                    l = weight * C.neighborhood_preseration(
                        pos, G, adj, 
                        k2i, i2k, 
                        degrees, maxDegree,
                        n_roots=sample_sizes[c], 
                        depth_limit=1)
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)

                elif c == 'crossings':
#                     loss += weight * C.crossings(
#                         pos, G, k2i, reg_coef=0.01, 
#                         niter=20, 
#                         sampleSize=sample_sizes['crossings'], sampleOn='edges'
#                     )


                    ## neural crossing detector
                    sample = self.sample(c)
                    sample = torch.stack(sample, 1)
                    edge_pair_pos = self.pos[sample].view(-1,8)
                    labels = utils.are_edge_pairs_crossed(edge_pair_pos)
                    
                    ## train crossing detector
                    self.crossing_detector.train()
                    preds = self.crossing_detector(edge_pair_pos.detach().to(device)).view(-1)
                    loss_nn = self.crossing_detector_loss_fn(preds, (labels.float()*0.8).to(device))
                    self.crossing_detector_optimizer.zero_grad()
                    loss_nn.backward()
                    self.crossing_detector_optimizer.step()
                    
                    ## loss of crossing
                    self.crossing_detector.eval()
                    preds = self.crossing_detector(edge_pair_pos.to(device)).view(-1)
                    
                    l = weight * self.crossing_pos_loss_fn(preds, (labels.float()*0.01).to(device))
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)
                    
                elif c == 'crossing_angle_maximization':
                    l = weight * C.crossing_angle_maximization(
                        pos, G, k2i, i2k, 
                        sampleSize=sample_sizes[c], 
#                         sampleOn='crossings') ## SLOW for large sample size
                        sampleOn='edges')
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)

                elif c == 'aspect_ratio':
                    l = weight * C.aspect_ratio(
                        pos, 
                        sampleSize=sample_sizes[c])
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)


                elif c == 'angular_resolution':
                    sample = [self.i2k[i.item()] for i in self.sample(c)]
                    l = weight * C.angular_resolution(
                        pos, G, k2i, 
                        sampleSize=sample_sizes[c],
                        sample=sample,
                    )
                    
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)


                elif c == 'vertex_resolution':
                    sample = self.sample(c)
                    l, vr_target_dist, vr_target_weight = C.vertex_resolution(
                        pos, 
                        sample=sample, 
                        target=1/len(G)**0.5, 
                        prev_target_dist=vr_target_dist,
                        prev_weight=vr_target_weight
                    )
                    l = weight * l
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)


                elif c == 'gabriel':
                    l = weight * C.gabriel(
                        pos, G, k2i, sampleSize=sample_sizes[c])
                    loss += l
                    self.grad_clamp(l, c, weight, optimizer, ref)


                else:
                    print(f'Criteria not supported: {c}')
#             if len(self.grads) > 0:
#                 pos.grad = sum(g for c,g in self.grads.items())
#                 pos.grad.clamp_(-grad_clamp, grad_clamp)
#                 optimizer.step()
#                 ref = pos.grad.norm(dim=1).max()

            optimizer.zero_grad()
            loss.backward()
            pos.grad.clamp_(-grad_clamp, grad_clamp)
            optimizer.step()
            
            
            
            self.runtime += time.time() - t0

            if vis_interval is not None and vis_interval>0 \
            and self.i%vis_interval==vis_interval-1:
                pos_numpy = pos.detach().cpu().numpy()
                pos_G = {k:pos_numpy[k2i[k]] for k in G.nodes}
                if display is not None and clear_output:
                    display.clear_output(wait=True)
                V.plot(
                    G, pos_G,
                    self.loss_curve, 
                    self.i, self.runtime, 
                    node_size=0,
                    edge=True, show=True, save=False
                )



#             if loss.isnan():
#                 raise Exception('loss is nan')
#                 break
            if pos.isnan().any():
                raise Exception('pos is nan')
                break

            if self.i % 100 == 99:
                iterBar.set_postfix({'loss': loss.item(), })    

            weighted_sum_of_loss = weighted_sum_of_loss*s + loss.item()
            sum_of_weight = sum_of_weight*s+1
            self.loss_curve.append(weighted_sum_of_loss/sum_of_weight)
                

                
            if scheduler is not None:
                scheduler.step(self.loss_curve[-1])


            lr = optimizer.param_groups[0]['lr']
            if lr <= scheduler.min_lrs[0]:
                break

                
            if (evaluate_interval is not None
                and evaluate_interval>0 
                and self.i%evaluate_interval == evaluate_interval-1
                or self.i == max_iter-1):
                
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
        return self.get_result_dict(evaluate, sample_sizes)
    
    
    def init_sampler(self, criteria_weights):
        self.samplers = {}
        self.dataloaders = {}
        for c,w in criteria_weights.items():
            if w == 0:
                continue
            if c == 'stress':
                self.dataloaders[c] = DataLoader(
                    self.node_index_pairs, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
            elif c == 'edge_uniformity':
                pass
#                 self.dataloaders[c] = DataLoader(self.edge_indices, batch_size=self.sample_sizes[c])
            elif c == 'neighborhood_preservation':
                pass
            elif c == 'crossings':
                self.dataloaders[c] = DataLoader(
                    self.non_incident_edge_pairs, 
                    batch_size=self.sample_sizes[c], 
                    shuffle=True)
            elif c == 'crossing_angle_maximization':
                pass
            elif c == 'aspect_ratio':
                pass
            elif c == 'angular_resolution':
                    ## TODO sampling all PAIRS of incident edges, instead of nodes
                self.dataloaders[c] = DataLoader(
                    range(len(self.G.nodes)), 
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
                
            elif c == 'vertex_resolution':
                self.dataloaders[c] = DataLoader(
                    self.node_index_pairs, 
                    batch_size=self.sample_sizes[c],
                    shuffle=True)
            elif c == 'gabriel':
                pass
    
    
    def sample(self, criterion):
        if criterion not in self.samplers:
            self.samplers[criterion] = iter(self.dataloaders[criterion])
        try:
            sample = next(self.samplers[criterion])
        except StopIteration:
            self.samplers[criterion] = iter(self.dataloaders[criterion])
            sample = next(self.samplers[criterion])
        return sample
        
    
    def get_result_dict(self, evaluate, sample_sizes):
        return dict(
            iter=self.i, 
            loss_curve=self.loss_curve,
            runtime=self.runtime,
            qualities_by_time=self.qualities_by_time,
            qualities=self.qualities_by_time[-1]['qualities'],
            sample_sizes=self.sample_sizes,
            pos=self.pos,
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
                qualityMeasures[q] = 1 - Q.neighborhood_preservation(
                    pos, self.G, self.adj, self.i2k)
            elif q == 'crossings':
                qualityMeasures[q] = Q.crossings(pos, self.edge_indices)
            elif q == 'crossing_angle_maximization':
                qualityMeasures[q] = Q.crossing_angle_maximization(
                    pos, self.G.edges, self.k2i)
            elif q == 'aspect_ratio':
                qualityMeasures[q] = 1 - Q.aspect_ratio(pos)
            elif q == 'angular_resolution':
                qualityMeasures[q] = 1- Q.angular_resolution(pos, self.G, self.k2i)
            elif q == 'vertex_resolution':
                qualityMeasures[q] = 1 - Q.vertex_resolution(pos, target=1/len(self.G)**0.5)
            elif q == 'gabriel':
                qualityMeasures[q] = 1 - Q.gabriel(pos, self.G, self.k2i)
           
            if verbose:
                print(f'done in {time.time()-t0:.2f}s')
        return qualityMeasures
    
    def save(self, fn='result.pkl'):
        with open(fn, 'wb') as f:
            pkl.dump(dict(
                G=self.G,
                pos=self.pos,
                i2k=self.i2k,
                k2i=self.k2i,
                iter=self.i,
                runtime=self.runtime,
                loss_curve=self.loss_curve,
                qualities_by_time = self.qualities_by_time,
            ), f)
        