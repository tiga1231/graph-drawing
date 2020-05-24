import networkx as nx
from input_functions import *

def generate_edges_for_complete_graphs(vertex_arr):
 edges = []
 for i in range(len(vertex_arr)):
  for j in range(i+1,len(vertex_arr)):
   edges.append((vertex_arr[i], vertex_arr[j]))
 return edges

def block_graph(nc, cs):
 '''
 nc is number of cluster
 cs is cluster size
 '''
 G = nx.Graph()
 edges = []
 for i in range(nc):
  edges += generate_edges_for_complete_graphs([j for j in range(i*cs,(i+1)*cs)])
  edges += [(i*cs,((i+1)*cs)%(nc*cs))]
 G.add_edges_from(edges)
 return G


# Gs = [nx.cycle_graph(10), nx.grid_graph(dim=[5,5]), nx.full_rary_tree(2, 15), nx.complete_graph(20), nx.complete_bipartite_graph(5, 5), nx.dodecahedral_graph(), nx.hypercube_graph(3), block_graph(5,5), build_networkx_graph('input9.txt')]
# for G in Gs:
#   networkx_to_json(G)

Gs = [
('cycle_graph_10', nx.cycle_graph(10)),
('grid_graph_5_5', nx.grid_graph(dim=[5,5])),
('grid_graph_10_6', nx.grid_graph(dim=[10,6])),
('full_rary_tree', nx.full_rary_tree(2, 15)), 
('complete_graph_20', nx.complete_graph(20)), 
('complete_bipartite_graph_5_5', nx.complete_bipartite_graph(5, 5)),
('dodecahedral_graph', nx.dodecahedral_graph()), 
('hypercube_graph_3', nx.hypercube_graph(3)),
('block_graph_5_5', block_graph(5,5)),
('input_9', build_networkx_graph('input9.txt')),
]
for name, G in Gs:
  print(name)
  networkx_to_json_2(G, name)