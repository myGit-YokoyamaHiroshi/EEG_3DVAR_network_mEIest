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
def Lin3dvar(ub,w,H,R,B):
    K  = B @ (H.T) @ np.linalg.inv(R + H@B@(H.T))
    ua = ub + K @ (w-H@ub) #solve a linear system
        
    return ua

def Lin3dvar_with_adaptive_error_cov_B(ub,w,H,R,B):
    I  = np.eye(len(ub))
    
    K  = B @ (H.T) @ np.linalg.inv(R + H@B@(H.T))
    ua = ub + K @ (w-H@ub) #solve a linear system
    #### iterative tuning algorithm 
    ####   for back ground and observation error covariance, B and R
    #### [Chapnik et al, Q. J. Roy. Meteorol. Soc, 2006]
    Bi = np.linalg.inv(B)
    Jb = (ua-ub).T @ Bi @ (ua-ub)
    sb = 2*Jb/np.trace(K@H)
    B  = sb*B
    
    Ri = np.linalg.inv(R)
    Jr = (w-H@ub).T @ Ri @ (w-H@ub)
    sr = 2*Jr/np.trace(I-K@H)
    R  = sr*R
    
    return ua, B, R

def do_3dvar(eeg, eeg_obs, param_true, K, alp,
             fs_dwn, fs, time, T, Ttr, Nch, Ntri, 
             seed=None):
    
    Nt      = int(fs*T)
    
    eeg     = eeg[:Nt, :]
    yobs    = eeg_obs[:Nt, :]
    time    = time[:Nt]
    
    # fig_dir = current_path + '/figures/est_results/Nroi%02d/Ttr%03dsec_fsobs%03dHz/tri%02d/'%(Nch, Ttr, fs_dwn, Ntri+1)
    # if os.path.exists(fig_dir)==False:  # Make the directory for data saving
    #     os.makedirs(fig_dir)
    
    if seed is not None:
        np.random.seed(seed)
        
    t_dwn   = np.arange(0, int(T*fs_dwn), 1)/fs_dwn
    y_dwn   = np.zeros((int(Ttr*fs_dwn), Nch))

    Nx      = int(Nch * 6)
    Nparam1 = 3 + 2*Nch
    Nparam2 = int(Nch*(Nch-1))
    Nstate  = int(Nx + Nparam1 + Nparam2)
    Nx      = int(Nch * 6)
    Nparam1 = 3 + 2*Nch
    Nparam2 = int(Nch*(Nch-1))
    Nstate  = int(Nx + Nparam1 + Nparam2)

    dt     = 1/fs
    dt_dwn = 1/fs_dwn
    sig_x  = dt/dt_dwn
    sig_p  = 1E-5#1E-4
    sig_xp = np.sqrt(1E-6)

    Bxx    = sig_x*np.eye(Nx)
    Bpp    = sig_p*np.eye(Nparam1 + Nparam2)
    Bxp    = np.random.normal(loc=0, scale=sig_xp, 
                              size=(Nx, Nparam1+Nparam2))**2
    Bpx    = Bxp.T
    Bm     = np.vstack((np.hstack((Bxx, Bxp)), np.hstack((Bpx, Bpp))))

    v0       = np.random.rand(Nx)
    A0       = 3.25 * np.ones(Nch)
    B0       =   22 * np.ones(Nch)
    a0       =  100 
    b0       =   50 
    p0       =  220
    K0       = np.random.uniform(low=.0, high=1, size=Nparam2)#
    param0   = np.hstack((A0, B0, a0, b0, p0, K0))

    va       = np.zeros([Nt, Nx])
    va[0,:]  = v0

    h        = np.array([0, 1, -1, 0, 0, 0])
    h        = np.kron(np.eye(Nch), h)
    H        = np.hstack((h, np.zeros((Nch, Nparam1 + Nparam2))))

    sig_y    = .1 # standard deviation for measurement noise
    R        = sig_y * np.eye(Nch)

    xpred      = np.hstack((v0, param0))
    ypred      = np.zeros(yobs.shape)
    ypred[0,:] = H@xpred

    param_pred = xpred[Nx:]
    cnt        = 0
    #%%
    for k in range(Nt-1):
        t_now = (k+1)*dt
        
        if t_now <= Ttr:
            param_pred = xpred[Nx:Nx+Nparam1]
            A_pred     = param_pred[:Nch]
            B_pred     = param_pred[Nch:2*Nch]
            a_pred     = param_pred[2*Nch]
            b_pred     = param_pred[2*Nch+1]
            p_pred     = param_pred[2*Nch+2]
            
            Jij_vec    = 0.5*np.tanh(10*(xpred[Nx+Nparam1:]-0.5)) + 0.5
            Jij_mat    = np.ones((Nch,Nch)) - np.eye(Nch)
            Jij_mat[Jij_mat==1] = Jij_vec
            
            xpred[Nx:Nx+Nparam1] = param_pred
            xpred[Nx+Nparam1:]   = Jij_vec
            

        va_mat     = va[k,:].reshape((6,Nch), order='f')
        va_mat     = runge_kutta(dt, func_JR_network_model, va_mat, 
                                 A_pred, a_pred, B_pred, b_pred, 
                                 p_pred, Jij_mat, alp)
        va[k+1,:]  = va_mat.reshape(-1, order='f') 
        
        xpred[:Nx] = va[k+1,:]
        
        
        if (np.mod(k+1, int(fs/fs_dwn))==0) & (t_now <= Ttr):
            y_dwn[cnt,:] = yobs[k,:]
            
            xpred,Bm,R = Lin3dvar_with_adaptive_error_cov_B(xpred, yobs[k,:], H, R, Bm)
            va[k+1,:]  = xpred[:Nx]
            
            cnt += 1
        
        ypred[k+1,:] = H@xpred
        param_pred = xpred[Nx:]
    #%%
    param_pred = xpred[Nx:Nx+Nparam1]
    Jij_vec    = xpred[Nx+Nparam1:]

    A_pred     = param_pred[:Nch]
    B_pred     = param_pred[Nch:2*Nch]
    a_pred     = param_pred[2*Nch]
    b_pred     = param_pred[2*Nch+1]
    p_pred     = param_pred[2*Nch+2]

    Jij_mat    = np.ones((Nch,Nch)) - np.eye(Nch)
    Jij_mat[Jij_mat==1] = Jij_vec
    #%%
    # seed_vis = 20
    # vmin     = 0
    # vmax     = .9
    # plt.figure(figsize=(7, 7))
    # _, pos = vis_directed_graph(K.T, vmin, vmax, seed_vis)
    # plt.title('exact')
    # plt.savefig(fig_dir + 'network_graph_exact.png', bbox_inches="tight")
    # plt.savefig(fig_dir + 'network_graph_exact.svg', bbox_inches="tight")
    # plt.show()

    # plt.figure(figsize=(7, 7))
    # vis_directed_graph(Jij_mat.T, vmin, vmax, seed_vis, pos=pos)
    # plt.title('estimated')
    # plt.savefig(fig_dir + 'network_graph_est.png', bbox_inches="tight")
    # plt.savefig(fig_dir + 'network_graph_est.svg', bbox_inches="tight")
    # plt.show()


    #%%
    # time_range = np.vstack((np.arange(Ttr/4, Ttr + Ttr/4, Ttr/4)-.5,
    #                         np.arange(Ttr/4, Ttr + Ttr/4, Ttr/4)+.5)).T
    # time_range = np.vstack((time_range,
    #                         np.array([T-1, T])))

    # for ch in range(Nch):
    #     fig = plt.figure(figsize=(15, 8))
    #     gs  = fig.add_gridspec(2,5)
    #     plt.subplots_adjust(wspace=0.5, hspace=0.6)
        
        
    #     ax1 = fig.add_subplot(gs[0, 0:5])
    #     ax1.plot(time, eeg[:,ch], c='#1f77b4', label='exact', 
    #               zorder=1, alpha=0.7, linewidth=1.5);
    #     ax1.plot(time, ypred[:,ch], color='#ff7f0e', label='predicted', 
    #               zorder=3, alpha=0.7, linewidth=1.5);
        
    #     plt.scatter(t_dwn[:int(Ttr*fs_dwn)], y_dwn[:,ch], label='observation', 
    #                 zorder=2, marker='.', c='k')
        
    #     [ax1.plot(time_range[i,:], [-0.8,-0.8], linewidth=3, c='k') for i in range(5)]
        
    #     ax1.axvspan(0, Ttr, color='gray', alpha=0.3, lw=0)
    #     ax1.set_title('EEG ch %02d'%ch)
    #     ax1.set_xlabel('time (s)')
    #     ax1.set_ylabel('amplitude (a.u.)')
    #     ax1.set_ylim(-1, 18)
    #     ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=26)
    #     ############
        
    #     for i in range(5):
    #         axn = fig.add_subplot(gs[1, i])
    #         axn.plot(time, eeg[:,ch], c='#1f77b4', label='exact', zorder=1, alpha=0.7, linewidth=1.5);
    #         axn.plot(time, ypred[:,ch], color='#ff7f0e', label='estimated', zorder=2, alpha=0.7, linewidth=1.5);
    #         plt.scatter(t_dwn[:int(Ttr*fs_dwn)], y_dwn[:,ch], marker='.', label='observation', c='k')
    #         axn.set_xlabel('time (s)')
    #         axn.set_xlim(time_range[i,:])
    #         axn.set_ylim(-1, 18)
    #         if i == 0:
    #             axn.set_ylabel('amplitude (a.u.)')
    #     #######
    #     plt.savefig(fig_dir + 'eeg_ch%02d.png'%ch, bbox_inches="tight")
    #     plt.savefig(fig_dir + 'eeg_ch%02d.svg'%ch, bbox_inches="tight")
    #     plt.show()
    #%%
    save_dir = current_path + '/save_data/est_results/Nroi%02d/Ttr%03dsec_fsobs%03dHz/'%(Nch, Ttr, fs_dwn)
    if os.path.exists(save_dir)==False:  # Make the directory for data saving
        os.makedirs(save_dir)
    
    save_dict = {}
    save_dict['A_pred']     = A_pred
    save_dict['B_pred']     = B_pred
    save_dict['a_pred']     = a_pred
    save_dict['b_pred']     = b_pred
    save_dict['p_pred']     = p_pred
    save_dict['Jij_pred']   = Jij_mat
    save_dict['eeg_pred']   = ypred
    save_dict['time']       = time
    
    
    save_dict['param_true'] = param_true
    save_dict['Jij_true']   = K
    save_dict['fs']         = fs
    save_dict['fs_dwn']     = fs_dwn
    save_dict['Ttr']        = Ttr
    save_dict['T']          = T
    
    
    save_name = 'results_%02d'%(Ntri+1)
    fullpath_save = save_dir + save_name 
    # np.save(fullpath_save, save_dict)
    np.savez_compressed(fullpath_save, save_dict)
    # return eeg, 
#%% Load generated synthetic observations
Nch = 10#
save_dir = current_path + '/save_data/synthetic_eeg/'
for file in os.listdir(save_dir):
    
    if 'Nch%02d'%Nch in file:
        split_str = os.path.splitext(file)
        name = split_str[0]
        ext  = split_str[1]
        break
    
data_path   = save_dir + name + ext
data_dict   = np.load(data_path, encoding='ASCII', allow_pickle='True').item()

eeg_obs_list    = data_dict['eeg_obs']
eeg_true_list   = data_dict['eeg_true']
# param_true_list = data_dict['param_true']
param_true      = data_dict['param_true']
K               = data_dict['K_true']
time            = data_dict['time']
fs              = data_dict['fs']
dt              = data_dict['dt']
Ntri            = data_dict['Ntri']
# T               = data_dict['T']
# Ttr             = data_dict['Ttr']

alp             = 1
del data_dict

#%% Apply 3d-var 
Ttr_list    = np.arange(20, 80, 20)
T_list      = Ttr_list + 20
fs_dwn_list = np.array([1000, 100, 5])

for i in range(len(Ttr_list)):
    Ttr = Ttr_list[i]
    T   = T_list[i]
    for j in range(len(fs_dwn_list)):
        fs_dwn = fs_dwn_list[j]
        np.random.seed(1000)
        Parallel(n_jobs=12, verbose=8)(delayed(do_3dvar)
                                       (eeg_true, eeg_obs, param_true,
                                        K, alp, fs_dwn, fs, time, T, Ttr, Nch, n_tri)
                                        for n_tri, (eeg_true, eeg_obs) 
                                             in enumerate(zip(eeg_true_list, eeg_obs_list)))

# #### for debug
# np.random.seed(1000)
# fs_dwn = 100
# T      = 80#200
# Ttr    = 60#180
# do_3dvar(eeg_true_list[0], eeg_obs_list[0], param_true_list[0],
#          K, alp, fs_dwn, fs, time, T, Ttr, Nch, 0)
