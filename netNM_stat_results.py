# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 01:07:54 2024

@author: H.Yokoyama
"""

from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
get_ipython().magic('clear')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

import numpy as np
# from neuralmass import *
from my_modules.neuralmassNET import Sigm, postsynaptic_potential_function
from my_modules.neuralmassNET import runge_kutta, func_JR_network_model
from my_modules.visualize_graph import vis_directed_graph
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 0.5
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 300 

import sys
sys.path.append(current_path)
#%%


def weighted_jaccard(x, y):
    q = np.concatenate([x,y], axis=1)
    return np.sum(np.amin(q,axis=1))/np.sum(np.amax(q,axis=1))

def PSI(x, y):
    from scipy import signal as sg
    x_phs = np.angle(sg.hilbert((x-x.mean(axis=0)).T).T)
    y_phs = np.angle(sg.hilbert((y-y.mean(axis=0)).T).T)
    
    PSI = abs(np.mean(np.exp(1j*(x_phs-y_phs)),axis=0))
    
    return PSI


def SHD(target, pred, double_for_anticausal=True):
    r"""Compute the Structural Hamming Distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either 
    missing or not in the target graph is counted as a mistake. Note that 
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the 
    `double_for_anticausal` argument accounts for this remark. Setting it to 
    `False` will count this as a single mistake.

    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of 
            ones and zeros.
        prediction (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two 
            mistakes. Default: True
 
    Returns:
        int: Structural Hamming Distance (int).

            The value tends to zero as the graphs tend to be identical.

    Examples:
        >>> from cdt.metrics import SHD
        >>> from numpy.random import randint
        >>> tar, pred = randint(2, size=(10, 10)), randint(2, size=(10, 10))
        >>> SHD(tar, pred, double_for_anticausal=False) 
    """                           

    diff = np.abs(target - pred)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff)/2
#%% Load generated synthetic observations
Nch      = 5#10#
dist     = 'SHD'
save_dir = current_path + '/save_data/synthetic_eeg/'
for file in os.listdir(save_dir):
    
    if 'Nch%02d'%Nch in file:
        split_str = os.path.splitext(file)
        name = split_str[0]
        ext  = split_str[1]
        break
    
data_path   = save_dir + name + ext
data_dict   = np.load(data_path, encoding='ASCII', allow_pickle='True').item()

eeg_true_list   = data_dict['eeg_true']
time            = data_dict['time']
fs              = data_dict['fs']
dt              = data_dict['dt']
param_true      = data_dict['param_true']

EI_true         = np.vstack((param_true[:Nch], param_true[Nch:2*Nch])).T
alp             = 1
del data_dict
#%%
fs           = 2000
save_dir     = current_path + '/save_data/est_results/Nroi%02d/'%Nch

# Ttr_list     = np.arange(20, 200, 20)
# T_list       = Ttr_list + 20
# fs_dwn_list  = np.array([1000, 500, 100, 50, 10])

Ttr_list    = np.arange(20, 80, 20)
T_list      = Ttr_list + 20
fs_dwn_list = np.array([1000, 100, 5])

Ntri         = 50

EI_pred_all  = np.zeros((Ntri, Nch, 2))
Jij_pred_all = np.zeros((Ntri, Nch,Nch, len(Ttr_list), len(fs_dwn_list)))  
Jij_sim      = np.zeros((Ntri, len(Ttr_list), len(fs_dwn_list)))   
eeg_psi      = np.zeros((Ntri, 10, len(Ttr_list), len(fs_dwn_list)))   
eeg_err      = np.zeros((Ntri, 10, len(Ttr_list), len(fs_dwn_list)))

for i, Ttr in enumerate(Ttr_list):
    for j, fs_dwn in enumerate(fs_dwn_list):
        data_path = save_dir + 'Ttr%03dsec_fsobs%03dHz/'%(Ttr, fs_dwn)
        
        for tri in range(Ntri):
            fname = 'results_%02d.npz'%(tri+1)
            # data_dict   = np.load(data_path + fname, 
            #                       encoding='ASCII', allow_pickle='True').item()
            data_dict   = np.load(data_path + fname, 
                                  encoding='ASCII', allow_pickle='True')['arr_0'].item()
            
            
            Jij_true = data_dict['Jij_true'] 
            Jij_pred = data_dict['Jij_pred'] 
            
            eeg_true = eeg_true_list[tri][:int(fs*(Ttr+20)), :]
            eeg_pred = data_dict['eeg_pred']
            
            
            Jij_pred_all[tri,:,:,i,j] = Jij_pred
            EI_pred_all[tri,:,0]      = data_dict['A_pred']
            EI_pred_all[tri,:,1]      = data_dict['B_pred']
            
            if dist == 'SHD':
                Jij_sim[tri,i,j] = SHD(Jij_true/np.linalg.norm(Jij_true), 
                                       Jij_pred/np.linalg.norm(Jij_pred), 
                                       double_for_anticausal=False)
            elif dist == 'cos':
                Jij_sim[tri,i,j] = cosine_similarity(Jij_true.reshape(-1)[np.newaxis,:],
                                                     Jij_pred.reshape(-1)[np.newaxis,:]).reshape(-1)[0]
            cnt = 0
            for t in range(0,20,2):
                dur      = int(fs*2)
                idx_st   = int(fs*(t))
                idx_en   = idx_st + dur
                
                eeg_psi[tri,cnt,i,j] = PSI(eeg_true[idx_st:idx_st+dur,:], 
                                       eeg_pred[idx_st:idx_st+dur,:]).mean()#np.sqrt(err_squred.mean())
                
                eeg_err[tri,cnt,i,j] = abs(eeg_true[idx_st:idx_st+dur,:]-eeg_pred[idx_st:idx_st+dur,:]).mean()
                
                cnt += 1
#%%
plt.rcParams["font.size"] = 18
fig_path = current_path + '/figures/est_results/Nroi%02d/'%(Nch)
if os.path.exists(fig_path)==False:  # Make the directory for data saving
    os.makedirs(fig_path)

vmax = .65
vmin = .55

Jij_sim_mean = np.median(Jij_sim, axis=0)

centers = [fs_dwn_list[0], fs_dwn_list[-1],
           Ttr_list[0],    Ttr_list[-1]]

dx,     = np.diff(centers[:2])/(Jij_sim_mean.shape[1]-1)
dy,     = -np.diff(centers[2:])/(Jij_sim_mean.shape[0]-1)
extent  = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]

if dist == 'cos':
    cmap = plt.cm.Blues
elif dist == 'SHD':
    cmap = plt.cm.Reds

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(Jij_sim_mean, 
               origin='lower', #vmin=vmin, vmax=vmax,  
               aspect=.5,#extent=extent,
               cmap=cmap,
               interpolation='hanning')
#### major ticks
ax.set_xticks(np.arange(0, len(fs_dwn_list)))
ax.set_xticklabels(['1/%d'%fs_dwn for fs_dwn in fs/fs_dwn_list], rotation=45)
ax.set_yticks(np.arange(0, len(Ttr_list)))
ax.set_yticklabels(['%d'%Ttr for Ttr in Ttr_list])

ax.set_xlabel('ratio $fs_{obs}~/~fs_{sim}$')
ax.set_ylabel('time length of training periods\n$T_{train}$ (s)')

#### color bar
cbar = fig.colorbar(im)

if dist == 'cos':
    cbar.ax.set_ylabel('graph similarity (a.u.)')
elif dist == 'SHD':
    cbar.ax.set_ylabel('structural hamming distance (a.u.)')
plt.savefig(fig_path + 'network_similarity_%s_Nch%02d.png'%(dist,Nch), bbox_inches="tight")
plt.savefig(fig_path + 'network_similarity_%s_Nch%02d.svg'%(dist,Nch), bbox_inches="tight")
plt.show()

#%%
vmax = .98
vmin = .97

eeg_psi_mean = np.median(eeg_psi, axis=0)

fig= plt.figure(figsize=(10,4))
gs  = fig.add_gridspec(1,4, width_ratios=[1,1,1,.1])
plt.subplots_adjust(wspace=.5)

cnt = 0
for t in [0, 4, 9]:
    ax = fig.add_subplot(gs[0, cnt])
    im = ax.imshow(eeg_psi_mean[t,:,:], 
                   origin='lower', vmin=vmin, vmax=vmax,  
                   aspect=.5,
                   cmap=plt.cm.Blues,
                   interpolation='hanning')
    #### major ticks
    ax.set_xticks(np.arange(0, len(fs_dwn_list)))
    ax.set_xticklabels(['1/%d'%fs_dwn for fs_dwn in fs/fs_dwn_list], rotation=45)
    ax.set_yticks(np.arange(0, len(Ttr_list)))
    ax.set_yticklabels(['%d'%Ttr for Ttr in Ttr_list])
    
    ax.set_xlabel('ratio $fs_{obs}~/~fs_{sim}$')
    
    if t==0:
        ax.set_ylabel('time length of training periods\n$T_{train}$ (s)')
    
    im_ratio = 4 / 6
    
    if t == 9:
        ax_cbar = fig.add_subplot(gs[0, 3])
        ax_cbar.set_axis_off()
        plt.colorbar(im, ax=ax_cbar, label='Phase coherence (a.u.)', 
                     fraction=im_ratio, pad=0.035)
    
    ax.set_title('test period\n %.1f-%.1f sec'%(2*t, 2*(t+1)))
    
    cnt  += 1

fig.suptitle('phase coherence between observations and predictions')
fig.tight_layout()
fig.subplots_adjust(top=1) 
plt.savefig(fig_path + 'phase_est_acc_Nch%02d.png'%(Nch), bbox_inches="tight")
plt.savefig(fig_path + 'phase_est_acc_Nch%02d.svg'%(Nch), bbox_inches="tight")
plt.show()

#%%
vmax = .5
vmin = .25

eeg_err_mean = np.median(eeg_err, axis=0)


fig= plt.figure(figsize=(10,4))
gs  = fig.add_gridspec(1,4, width_ratios=[1,1,1,.1])
plt.subplots_adjust(wspace=.5)

cnt = 0
for t in [0, 4, 9]:
    ax = fig.add_subplot(gs[0, cnt])
    im = ax.imshow(eeg_err_mean[t,:,:], 
                   origin='lower', vmin=vmin, vmax=vmax,  
                   aspect=.5,
                   cmap=plt.cm.Reds,
                   interpolation='hanning')
    #### major ticks
    ax.set_xticks(np.arange(0, len(fs_dwn_list)))
    ax.set_xticklabels(['1/%d'%fs_dwn for fs_dwn in fs/fs_dwn_list], rotation=45)
    ax.set_yticks(np.arange(0, len(Ttr_list)))
    ax.set_yticklabels(['%d'%Ttr for Ttr in Ttr_list])
    
    ax.set_xlabel('ratio $fs_{obs}~/~fs_{sim}$')
    
    if t==0:
        ax.set_ylabel('time length of training period\n$T_{train}$ (s)')
    
    im_ratio = 4 / 6
    
    
    if t == 9:
        ax_cbar = fig.add_subplot(gs[0, 3])
        ax_cbar.set_axis_off()
        plt.colorbar(im, ax=ax_cbar, label='mean absolute error (a.u.)', 
                     fraction=im_ratio, pad=0.035)
    ax.set_title('test period\n %.1f-%.1f sec'%(2*t, 2*(t+1)))
    
    cnt  += 1
    # #### color bar
    # cbar = fig.colorbar(im)
    # # cbar.set_ticks(np.arange(0.5, vmax))
    # cbar.ax.set_ylabel('mean absolute error (a.u.)')
fig.suptitle('Prediction error of observation amplitude')
fig.tight_layout()
fig.subplots_adjust(top=1.) 
plt.savefig(fig_path + 'eeg_est_acc_Nch%02d.png'%(Nch), bbox_inches="tight")
plt.savefig(fig_path + 'eeg_est_acc_Nch%02d.svg'%(Nch), bbox_inches="tight")
plt.show()
#%%
Jij_pred_mean = np.median(Jij_pred_all,axis=0)


fig= plt.figure(figsize=(10,5))
gs  = fig.add_gridspec(1,3, width_ratios=[1,1,.1])
plt.subplots_adjust(wspace=.5)

seed_vis = 20; 
ax = fig.add_subplot(gs[0, 0])
_, pos = vis_directed_graph(Jij_true.T, 0, 1, seed_vis)
plt.title('exact')

cbar_ax = fig.add_subplot(gs[0, 2])

if dist == 'cos':
    rows, cols  = np.where(Jij_sim_mean==Jij_sim_mean.max())
elif dist=='SHD':
    rows, cols  = np.where(Jij_sim_mean==Jij_sim_mean.min())
print('Ttrain = %dsec, fs ratio = 1/%d'%(Ttr_list[rows[0]],
                                         fs/fs_dwn_list[cols[0]]))
    
ax = fig.add_subplot(gs[0, 1])
vis_directed_graph(Jij_pred_mean[:,:,rows[0], cols[0]].T, 0, 1, seed_vis, 
                   pos=pos, cbar_ax=cbar_ax);
plt.title('estimated \n(trial average: median)')
# if dist=='cos':
#     tri, rows, cols  = np.where(Jij_sim==Jij_sim.max())
# elif dist=='SHD':
#     tri, rows, cols  = np.where(Jij_sim==Jij_sim.min())
# print('trial %02d, Ttrain = %dsec, fs ratio = 1/%d'%(tri+1, 
#                                                    Ttr_list[rows[0]],
#                                                    fs/fs_dwn_list[cols[0]]))

# ax = fig.add_subplot(gs[0, 1])
# vis_directed_graph(Jij_pred_all[tri[0],:,:,rows[0], cols[0]].T, 0, 1, seed_vis, 
#                    pos=pos, cbar_ax=cbar_ax);

plt.savefig(fig_path + 'network_est_result_Nch%02d.png'%(Nch), bbox_inches="tight")
plt.savefig(fig_path + 'network_est_result_Nch%02d.svg'%(Nch), bbox_inches="tight")

plt.show()
