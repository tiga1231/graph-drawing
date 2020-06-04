#in this file, all input related functions are added

import networkx as nx
import random
import json
import torch

# this functions takes two arguments
# graph: this is a networkx graph
# file_name: this is the output file
# this functions write the graph in the file in txt format
# later we can read this file using the functions defined later
# note that, the positions are selected in a random way
def write_networx_graph(graph, file_name):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(random.randint(1,300))+" "+str(random.randint(1,300))+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def is_comment(x):
 if x[0]=='#':
  return True
 return False

def take_input(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 n = int(l)
 coord_list = list()
 for i in range(n):
  while True:
   l = file.readline()
   if not is_comment(l):
    break
  t_arr1 = []
  t_arr2 = l.split()
  t_arr1.append(float(t_arr2[0]))
  t_arr1.append(float(t_arr2[1]))
  coord_list.append(t_arr1)
 edge_list = list()
 for i in range(n*n):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<2):break
    t_arr1.append(int(t_arr2[0]))
    t_arr1.append(int(t_arr2[1]))
    edge_list.append(t_arr1)
 m = len(edge_list)

 matrix = [[0] * n for i in range(n)]

 for [u, v] in edge_list:
    matrix[u][v] = matrix[v][u] = 1

 file.close()
 return n, coord_list, edge_list

def take_input_grid_size(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 n = int(l)
 coord_list = list()
 for i in range(n):
  while True:
   l = file.readline()
   if not is_comment(l):
    break
  t_arr1 = []
  t_arr2 = l.split()
  t_arr1.append(float(t_arr2[0]))
  t_arr1.append(float(t_arr2[1]))
  coord_list.append(t_arr1)
 edge_list = list()
 width = 0
 height = 0
 for i in range(n*n):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<2):
      width = int(t_arr2[0])
      l = file.readline()
      height = int(l.split()[0])
      break
    t_arr1.append(int(t_arr2[0]))
    t_arr1.append(int(t_arr2[1]))
    edge_list.append(t_arr1)
 m = len(edge_list)

 matrix = [[0] * n for i in range(n)]

 for [u, v] in edge_list:
    matrix[u][v] = matrix[v][u] = 1

 file.close()
 return n, coord_list, edge_list, width, height


import json

def take_input_from_json(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 graph = json.loads(file.read())
 #print(graph)

 n = len(graph['nodes'])
 coord_list = list()
 edge_list = list()
 matrix = [[0] * n for i in range(n)]

 for v in graph['nodes']:
  arr = []
  arr.append(float(v['x']))
  arr.append(float(v['y']))
  coord_list.append(arr)

 for e in graph['edges']:
  matrix[e['target']][e['source']] = matrix[e['source']][e['target']] = 1
  t_arr1 = []
  t_arr1.append(e['source'])
  t_arr1.append(e['target'])
  edge_list.append(t_arr1)

 file.close()
 return n, coord_list, edge_list

def take_input_from_json_grid_size(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 graph = json.loads(file.read())
 #print(graph)

 n = len(graph['nodes'])
 coord_list = list()
 edge_list = list()
 matrix = [[0] * n for i in range(n)]

 for v in graph['nodes']:
  arr = []
  arr.append(float(v['x']))
  arr.append(float(v['y']))
  coord_list.append(arr)

 for e in graph['edges']:
  matrix[e['target']][e['source']] = matrix[e['source']][e['target']] = 1
  t_arr1 = []
  t_arr1.append(e['source'])
  t_arr1.append(e['target'])
  edge_list.append(t_arr1)

 width = 1000000
 height = 1000000
 if "width" in graph.keys():
  width = graph['width']
 if "height" in graph.keys():
  height = graph['height']

 file.close()
 return n, coord_list, edge_list, width, height

def txt_to_json(input_file, output_file):
 n, coord_list, edge_list = take_input(input_file)
 graph = {}
 nodes = []
 max_x = 0
 max_y = 0
 for i in range(len(coord_list)):
  node = {}
  node['id'] = i
  node['x'] = coord_list[i][0]
  if max_x<coord_list[i][0]:max_x=coord_list[i][0]
  node['y'] = coord_list[i][1]
  if max_y<coord_list[i][1]:max_y=coord_list[i][1]
  nodes.append(node)
 graph['nodes'] = nodes
 edges = []
 for i in range(len(edge_list)):
  edge = {}
  edge['source']=edge_list[i][0]
  edge['target']=edge_list[i][1]
  edges.append(edge)
 graph['edges'] = edges
 graph['xdimension'] = max_x
 graph['ydimension'] = max_y
 with open(output_file, 'w') as outfile:
  json.dump(graph, outfile)

def txt_to_json_grid_size(input_file, output_file, width, height):
 n, coord_list, edge_list = take_input(input_file)
 graph = {}
 nodes = []
 max_x = 0
 max_y = 0
 for i in range(len(coord_list)):
  node = {}
  node['id'] = i
  node['x'] = coord_list[i][0]
  if max_x<coord_list[i][0]:max_x=coord_list[i][0]
  node['y'] = coord_list[i][1]
  if max_y<coord_list[i][1]:max_y=coord_list[i][1]
  nodes.append(node)
 graph['nodes'] = nodes
 edges = []
 for i in range(len(edge_list)):
  edge = {}
  edge['source']=edge_list[i][0]
  edge['target']=edge_list[i][1]
  edges.append(edge)
 graph['edges'] = edges
 graph['width'] = width
 graph['height'] = height
 with open(output_file, 'w') as outfile:
  json.dump(graph, outfile)


def json_to_txt(input_file, output_file):
 n, coord_list, edge_list = take_input_from_json(input_file)
 file = open(output_file,"w")
 file.write(str(n)+"\n");
 for j in range(n):
  file.write(str(coord_list[j][0])+" "+str(coord_list[j][1])+"\n")
 for e in edge_list:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def json_to_txt_grid_size(input_file, output_file):
 n, coord_list, edge_list, width, height = take_input_from_json_grid_size(input_file)
 file = open(output_file,"w")
 file.write(str(n)+"\n");
 for j in range(n):
  file.write(str(coord_list[j][0])+" "+str(coord_list[j][1])+"\n")
 for e in edge_list:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.write(str(width)+"\n")
 file.write(str(height)+"\n")
 file.close()


# this function directly builds a networkx graph from a txt file
def build_networkx_graph(filename):
 n, coord_list, edge_list = take_input(filename)
 G=nx.Graph()
 for e in edge_list:
  G.add_edge(e[0], e[1])
 return G

# this function directly builds a networkx graph from a txt file
def build_networkx_graph_grid_size(filename):
 n, coord_list, edge_list, width, height = take_input_grid_size(filename)
 G=nx.Graph()
 for e in edge_list:
  G.add_edge(e[0], e[1])
 return G, width, height

def build_directed_networkx_graph(filename):
 n, coord_list, edge_list = take_input(filename)
 G=nx.DiGraph()
 for e in edge_list:
  G.add_edge(e[0], e[1])
 return G

def parse_dot_file(file_name):
 file = open(file_name, 'r')
 arr = file.read().split(';')
 nodes = []
 edges = []
 node_coords = []
 edge_list = []
 for i in range(len(arr)):
 #for i in range(10):
  arr[i] = arr[i].strip()
  elmnt = arr[i].split()
  if len(elmnt)>2:
   if elmnt[1][0]=='[':
    #print(elmnt[0])
    nodes.append(elmnt)
   else:
    #print(elmnt[0]+elmnt[1]+elmnt[2])
    edges.append(elmnt)
  #print(elmnt)
 for i in range(len(nodes)):
  #print nodes[i]
  #print(nodes[i][0])
  coords = nodes[i][2][5:len(nodes[i][2])-2].split(',')
  for k in range(len(coords)):
   coords[k] = float(coords[k])
  node_coords.append(coords)
 for i in range(1,len(edges)):
  edg = []
  #print(edges[i])
  edg.append(int(edges[i][0]))
  edg.append(int(edges[i][2]))
  edge_list.append(edg)
 file.close()

 #print(node_coords)
 #print(edge_list)
 return node_coords, edge_list

def dot_to_json_grid_size(input_file, output_file, input_width, input_height):
 coord_list, edge_list = read_dot_file(input_file)
 #print(edge_list)
 graph = {}
 nodes = []
 max_x = 0
 max_y = 0
 for i in range(len(coord_list)):
  node = {}
  node['id'] = i
  node['x'] = int(coord_list[i][0])
  if max_x<coord_list[i][0]:max_x=coord_list[i][0]
  node['y'] = int(coord_list[i][1])
  if max_y<coord_list[i][1]:max_y=coord_list[i][1]
  nodes.append(node)
 graph['nodes'] = nodes
 edges = []
 for i in range(len(edge_list)):
  edge = {}
  edge['source']=edge_list[i][0]
  edge['target']=edge_list[i][1]
  edges.append(edge)
 graph['edges'] = edges
 graph['width'] = input_width
 graph['height'] = input_height
 #print(graph)
 with open(output_file, 'w') as outfile:
  json.dump(graph, outfile)

def read_dot_file(file_name):
 import networkx as nx
 import pygraphviz as pgv
 from networkx.drawing.nx_agraph import read_dot as nx_read_dot
 G = nx_read_dot(file_name)
 #print(G.edges())
 G = nx.DiGraph(G)
 #print(G.edges())
 node_coords = []
 for i in range(G.number_of_nodes()):
  coords = [float(c) for c in G.nodes[str(i)]['pos'].split(',')]
  node_coords.append(coords)
 edge_list = []
 for e in G.edges():
  u, v = e
  edge_list.append([int(u), int(v)])
 #print(edge_list)
 return node_coords, edge_list

def read_dot_file_with_arbitrary_node_id(file_name):
 import networkx as nx
 import pygraphviz as pgv
 from networkx.drawing.nx_agraph import read_dot as nx_read_dot
 G_labeled = nx_read_dot(file_name)
 #print(G.edges())
 G_labeled = nx.Graph(G_labeled)
 #print(G.edges())
 node_coords = []
 label_to_id = dict()
 id_to_label = dict()
 id_count = 0
 for v in G_labeled.nodes():
  label_to_id[v] = id_count
  id_to_label[id_count] = v
  id_count += 1
 G = nx.Graph()
 for u, v in G_labeled.edges():
  G.add_edge(label_to_id[u], label_to_id[v])
 for i in range(G.number_of_nodes()):
  coords = [float(c) for c in G_labeled.nodes[id_to_label[i]]['pos'].split(',')]
  node_coords.append(coords)
 edge_list = []
 for e in G.edges():
  u, v = e
  edge_list.append([int(u), int(v)])
 #print(edge_list)
 return node_coords, edge_list, label_to_id, id_to_label

def read_dot_file_with_label(file_name):
 import networkx as nx
 import pygraphviz as pgv
 from networkx.drawing.nx_agraph import read_dot as nx_read_dot
 G = nx_read_dot(file_name)
 #print(G.edges())
 G = nx.DiGraph(G)
 #print(G.edges())
 mp = nx.get_node_attributes(G, 'label')
 node_coords = []
 for i in range(G.number_of_nodes()):
  coords = [float(c) for c in G.nodes[str(i)]['pos'].split(',')]
  node_coords.append(coords)
 edge_list = []
 for e in G.edges():
  u, v = e
  edge_list.append([int(u), int(v)])
 #print(edge_list)
 return node_coords, edge_list, mp

def write_dot_file(file_name, coord_list, edge_list):
 import pygraphviz as pgv
 from networkx.drawing.nx_agraph import write_dot
 #print(edge_list)
 G = nx.DiGraph()
 pos = dict()
 n = len(coord_list)
 for j in range(n):
  #pos[j] = (coord_list[j][0], coord_list[j][1])
  pos[j] = str(coord_list[j][0]) + "," + str(coord_list[j][1])
 for e in edge_list:
  G.add_edge(e[0], e[1])
 for n in pos.keys():
  G.node[n]['pos'] = pos[n]
 #print(G.edges())
 write_dot(G, file_name)

def take_input_from_dot(input_file):
 node_coords, edge_list = parse_dot_file(input_file)
 n = len(node_coords)
 return n, node_coords, edge_list 


def take_input_force_directed(file_name):
 file = open(file_name, 'r')
 arr = file.read().split('\n')
 x = []
 y = []
 for i in arr[0].split(','):
  x.append(float(i))
 for i in arr[1].split(','):
  y.append(float(i))
 file.close()
 return x,y

def write_as_txt_random_position(file_name, graph):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(random.randint(1,300))+" "+str(random.randint(1,300))+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def write_as_txt(file_name, graph, x, y):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(x[j])+" "+str(y[j])+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def write_as_txt(file_name, edge_list, x, y):
 file = open(file_name,"w")
 file.write(str(len(x))+"\n");
 x_min = x[0]
 y_min = y[0]
 for j in range(len(x)):
  if x_min>x[j]:
   x_min = x[j]
  if y_min>y[j]:
   y_min = y[j]
 for j in range(len(x)):
  x[j] = x[j]-x_min
  y[j] = y[j]-y_min
 for j in range(len(x)):
  #print(x[j], y[j])
  #file.write(str(int(x[j]))+" "+str(int(y[j]))+"\n")
  file.write(str(x[j])+" "+str(y[j])+"\n")
 for e in edge_list:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def write_as_txt_grid_size(file_name, edge_list, x, y, width, height):
 file = open(file_name,"w")
 file.write(str(len(x))+"\n");
 x_min = x[0]
 y_min = y[0]
 for j in range(len(x)):
  if x_min>x[j]:
   x_min = x[j]
  if y_min>y[j]:
   y_min = y[j]
 for j in range(len(x)):
  x[j] = x[j]-x_min
  y[j] = y[j]-y_min
 for j in range(len(x)):
  #print(x[j], y[j])
  #file.write(str(int(x[j]))+" "+str(int(y[j]))+"\n")
  file.write(str(x[j])+" "+str(y[j])+"\n")
 for e in edge_list:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.write(str(width)+"\n")
 file.write(str(height)+"\n")
 file.close()

def networkx_to_json(G):
 graph = {}
 nodes = []
 max_x = 0
 max_y = 0
 for i in range(len(G.nodes())):
  node = {}
  node['id'] = i
  nodes.append(node)
 graph['nodes'] = nodes
 edges = []
 i = 0
 for e in G.edges():
  u, v = e
  edge = {}
  edge['source']=u
  edge['target']=v
  edges.append(edge)
  i += 1
 graph['edges'] = edges
 graph['xdimension'] = max_x
 graph['ydimension'] = max_y
 print(graph)



def dict2tensor(d, fill=None):
  n = len(d.keys())
  k2i = {k:i for i,k in enumerate(d.keys())}
  res = torch.zeros(len(d.keys()), len(d.keys()), device='cpu')
  for src_node, dst_nodes in d.items():
    for dst_node, distance in dst_nodes.items():
      if fill is not None:
        res[k2i[src_node],k2i[dst_node]] = fill
      else:
        res[k2i[src_node],k2i[dst_node]] = distance
  return res, k2i

def networkx_to_json_2(G, name):
  D, k2i = dict2tensor(dict(nx.all_pairs_shortest_path_length(G)))
  print(D.shape)
  n = len(G.nodes)
  eye = torch.eye(n)
  W = 1/(D**2+eye)

  graph = {
    'nodes': [
        {
            'index':i, 
            'id': str(n),
        } 
        for i, n in enumerate(G.nodes)
    ],
    'edges': [
        {
            'source': str(e1), 
            'target': str(e2)
        } 
        for e1,e2 in G.edges
    ],
    'weight': W.numpy().tolist(),
    'graphDistance': D.numpy().tolist(),
    'initPositions': None
  }
  
  with open(f'output/{name}.json', 'w') as f:
    json.dump(graph, f, indent=2)
