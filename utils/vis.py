## vis
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import mplot3d
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

import networkx as nx
import numpy as np

# def criterion_to_title(criterion):
#     return ' '.join([x.capitalize() for x in criterion.split('_')])


def colorScale2cmap(domain, range1):
    domain = np.array(domain)
    domain = (domain-domain.min())/(domain.max()-domain.min())
    range1 = np.array(range1)/255.0
    red = [r[0] for r in range1]
    green = [r[1] for r in range1]
    blue = [r[2] for r in range1]
    red = tuple((d,r,r) for d,r in zip(domain, red))
    green = tuple((d,r,r) for d,r in zip(domain, green))
    blue = tuple((d,r,r) for d,r in zip(domain, blue))
    return LinearSegmentedColormap('asdasdas', {'red':red, 'green': green, 'blue':blue})
    
    
def plot_weight(criteria_weights, max_iter, ax=None):
    
    t = np.linspace(0, max_iter, 250)
    if ax is None:
        plt.figure(figsize=[12,4])
        ax = plt.subplot(111)
        
    for i,[c,w] in enumerate(criteria_weights.items()):
        if callable(w):
            y = [w(ti) for ti in t]
        else:
            y = [w for ti in t]
        ax.plot(
            t,y,
            (['-','--'])[i%2], 
            label=c,
            alpha=1.0,
            lw=1.5
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Criteria weight')
    # ax.set_xlim(plt.xlim()[0], plt.xlim()[1]*1.1)
    ax.set_ylim(-0.1, plt.ylim()[1]*1.5)
    ax.legend()
    
    
def draw_graph_3d(ax, x, G, grad=None, alpha=0.1):
    ax.scatter(x[:,0], x[:,1], x[:,2])
    # ax.view_init(elev=20.0, azim=0)
    edgeLines = [(x[k2i[e0]][:3], x[k2i[e1]][:3]) for e0,e1 in G.edges]
    lc = Line3DCollection(edgeLines, linewidths=1, alpha=alpha)
    ax.add_collection(lc)
    if grad is not None:
        ax.quiver(x[:,0], x[:,1], x[:,2], 
                 -grad[:,0], -grad[:,1], -grad[:,2], length=4, colors='C1')
    return ax


def draw_graph(G, pos, ax, xlabel=None, ylabel=None, node_size=0, edge_width=1):
    # nx.draw_networkx_edges(G, pos=pos, ax=ax, width=edge_width)
    line_segments = LineCollection(
        [(pos[e0],pos[e1]) for e0,e1 in G.edges], 
        linewidth=edge_width, 
        color='#111'
        )
    ax.add_collection(line_segments)

    if node_size>0:
        nx.draw_networkx_nodes(G, 
            pos=pos, 
            node_size=node_size,
            # font_color='none',
            ax=ax
        )
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    
def plot(G, pos, lossHistory, i, totalTime, 
        criteria_weights, max_iter,
        grad=None, 
        node_size=0, 
        edge_width=1,
         # show=False, 
         # save=True, saveName='output.png',  
        title=None):
    
    fig = plt.figure(figsize=[24,8])

    ## graph
    ax = plt.subplot(121)
    if edge_width > 0:
        draw_graph(G, pos, ax=ax, xlabel=None, ylabel=None, node_size=node_size, edge_width=edge_width)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if grad is not None:
        plt.quiver(x[:,0], x[:,1], 
                   -grad[:,0], -grad[:,1], 
                   units='inches', label=f'neg grad (max={np.linalg.norm(grad, axis=1).max():.2e})')
        plt.legend()
    plt.axis('equal')
#     else:
#         ax = fig.add_subplot(1,2,1, projection='3d')
#         ax = draw_graph_3d(ax, x, G, grad, alpha=0.01)
    
    if title is None:
        plt.title(f'Iter: {i}, time: {totalTime:.2f}s'.format(i))
    else:
        plt.title(title)
        
    ## loss
    plt.subplot(222)
    plt.plot(lossHistory[0], lossHistory[1])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    if lossHistory:
        plt.title(f'Final Loss: {lossHistory[1][-1]:.4f}')
    
    ax = plt.subplot(224)
    plot_weight(criteria_weights, max_iter, ax)
    ## Lr
#     plt.subplot(224)
#     plt.plot(lrHistory)
#     plt.xlabel('Epoch')
#     plt.ylabel('LR')

    # if save:
    #     plt.savefig(saveName)
    
    # if show:
    #     plt.show()
    # else:
    #     plt.close()
    #     
    