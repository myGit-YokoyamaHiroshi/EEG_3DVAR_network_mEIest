# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:50:41 2024

@author: H.Yokoyama
"""
import numpy as np
import networkx as nx
import matplotlib.pylab as plt
from copy import deepcopy

#%% graph visualization
def vis_directed_graph(K, vmin, vmax, seed, pos=None, node_size_weight=None, cbar_ax=None, cbar_plot=False):
    import matplotlib as mpl
    
    if (cbar_ax is not None) & (cbar_plot==False):
        cbar_plot=True
    
    im_ratio = 5 / 6
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]
    
    G      = nx.from_numpy_array(K, create_using=nx.MultiDiGraph())
    # G.edges(data=True)
    
    sorted(G.edges(data=True), key=lambda edge: edge[2].get('weight', 1))
    
    
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
    
    labels = {i : i for i in G.nodes()}          
    
    node_sizes  = [1000  for i in range(len(G))]
    if node_size_weight is not None:
        node_sizes  = node_size_weight * node_sizes 
        
    M           = G.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1]  = 1
    edge_alphas[edge_alphas<.1] = .1
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.09',
                                         arrowsize=10, edge_color=edge_colors,
                                         width=4,
                                         edge_vmin=vmin, edge_vmax=vmax,
                                         alpha=edge_alphas)
    
    
    
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w')
    plt.axis('equal')
    

    # set alpha value for each edge
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
        
    for i in range(M):
        if vmin < 0:
            c_idx = int((weight[i]/vmax + 1)/2 * cm.N)
        elif vmin>=0:
            c_idx = int((edge_alphas[i] * cm.N))
            
        rgb = np.array(cm(c_idx))[0:3]
        # edges[i].set_alpha(edge_alphas[i])
        edges[i].set_color(rgb)
    
    ax = plt.gca()
    ax.set_axis_off()
    
    if cbar_plot==True:
        if cbar_ax is None:
            cbar_ax  = ax
            im_ratio = 0.05*im_ratio
        
        pc = mpl.collections.PatchCollection(edges, cmap=cm)
        pc.set_array(edge_colors)
        pc.set_clim(vmin=vmin, vmax=vmax)
        
        cbar_ax.set_axis_off()
        plt.colorbar(pc, ax=cbar_ax, label='coupling strength (a.u.)', 
                     fraction=im_ratio, pad=0.035)
                     #fraction=0.05*im_ratio, pad=0.035)
    
    return edges, pos