## custom
from utils import utils, vis
import criteria as C
import quality as Q
from gd2 import GD2


## third party
# from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import networkx as nx
from PIL import Image
from natsort import natsorted


## sys
from collections import defaultdict
from glob import glob
import json
import math
import random
import sys
import time


## numeric
import numpy as np
import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F

## vis
# import matplotlib.pyplot as plt


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'
# plt.style.use('ggplot')
# plt.style.use('seaborn-colorblind')




print('Opening JSON file..')
with open(sys.argv[1]) as json_file:
    input_param = json.load(json_file)


graph_str = (
    input_param['graph']
    + ' ' 
    + ' '.join(input_param[k] for k in natsorted(input_param) if k.startswith('graph_param_'))
)
metrics = input_param['metrics']
print(f'fn: {sys.argv[1].split("/")[-1]}')
print(f'metrics: {metrics}')
print(f'graph: {graph_str}')


print('generating graph', end=' ')
if input_param["graph"]=="tree":
    G = nx.balanced_tree(int(input_param["graph_param_1"]), int(input_param["graph_param_2"]))
elif input_param["graph"]=="hypercube":
    G = nx.hypercube_graph(int(input_param["graph_param_1"]))
elif input_param["graph"]=="grid":
    dim = int(input_param["graph_param_1"])
    G = nx.grid_graph(dim = [dim, dim])
print(f'of {len(G)} nodes')



max_iter = 20000
criteria_weights_default = {
    'stress': 4,
    'edge_uniformity':1,
    'neighborhood_preservation':0.5,
    'crossings':1,
    'crossing_angle_maximization':0.1,
    'aspect_ratio':10,
    'angular_resolution':0.01,
    'vertex_resolution':1,
    'gabriel': 0.01,
}
sample_sizes={
    'stress': 64,
    'edge_uniformity':10,
    'neighborhood_preservation':16,
    'crossings':10,
    'crossing_angle_maximization':10,
    'aspect_ratio': 'full',
    'angular_resolution':32,
    'vertex_resolution':int(len(G)**0.5),
    'gabriel':10,
},

## use default weights when we choose to optimize certain criteria
criteria_weights = {}
for m in input_param['metrics']:
    criteria_weights[m] = criteria_weights_default[m]


gd = GD2(G)
result = gd.optimize(
    criteria_weights=criteria_weights,
    sample_sizes=sample_sizes_default,
    
    evaluate='all',
#     evaluate={'neighborhood_preservation'},
    # evaluate=set(input_param['metrics']),
    
    max_iter=max_iter, 
    evaluate_interval=-1, ## only at the end
    vis_interval=-1, ## never plot
    
    optimizer_kwargs = dict(lr=1),
    scheduler_kwargs = dict(verbose=False),
)



for q,v in result['qualities'].items():
    print(q,v)

return_dict = dict(
    metric_value=result['qualities'],
    pos=result['pos'].detach().cpu().tolist(),
    time=result['runtime']
)

with open(input_param["output_file"], 'w') as fp:
    json.dump(return_dict, fp)



## draw graph for debugging
# pos = gd.pos.detach().numpy()
# pos_G = {k:pos[gd.k2i[k]] for k in gd.G.nodes}
# vis.plot(
#     gd.G, pos_G,
#     gd.loss_curve, 
#     result['iter'], result['runtime'],
#     edge=True, show=True, save=False
# )