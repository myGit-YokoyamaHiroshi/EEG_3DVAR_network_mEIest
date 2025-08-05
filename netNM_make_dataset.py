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

import scipy as sp
import sys
sys.path.append(current_path)

#%%
def generate_eeg(fs, dt, Nch, Nt, T_spurious, K, alp, 
                 A, a, B, b, p_m):
    
    p        = np.random.normal(loc=p_m, scale= 10, size=(Nt,Nch))
    
    time     = np.arange(0,Nt,1)/fs
    v        = np.zeros((6, Nch, Nt))
    v[:,:,0] = np.random.rand(6,Nch)

    for i in range(1, Nt):        
        v_now    = v[:, :, i-1]
        v_next   = runge_kutta(dt, func_JR_network_model, 
                              v_now, A, a, B, b, p[i,:], K, alp)
        v[:,:,i] = v_next

    eeg  = (v[1,:,T_spurious:] - v[2,:,T_spurious:]).T
    Nt   = Nt - T_spurious
    time = time[:Nt]
    v    = v[:,:,T_spurious:]
    ######
    SNRdB      = 10#
    sig_power  = np.mean((eeg-eeg.mean())**2)
    n_power    = sig_power/(10**(SNRdB/10));
    n_sig      = np.sqrt(n_power) * np.random.randn(Nt, Nch);
    yobs       = eeg + n_sig

    param_true = np.hstack((A,B,a, b,p_m))
    
    return eeg, yobs, param_true, time
#%%
save_dir = current_path + '/save_data/synthetic_eeg/'
if os.path.exists(save_dir)==False:  # Make the directory for data saving
    os.makedirs(save_dir)
#%% Generate synthetic EEGs with neural mass model
np.random.seed(1000)

Ntri = 50
fs   = 2000
dt   = 1/fs

# T    = 10#80 # s
# Ttr  = 6#60 # s

T    = 200#80 # s
Ttr  = 180#60 # s

T_spurious  = int(5*fs)
# Nch  = 5
Nch  = 10

Nt   = int(T*fs) + T_spurious
alp  = 1

if Nch == 5:
    #############################################    
    K    = np.array([[ 0, .0, .9, .9, .9],
                     [.0, .0, .9, .0, .0],
                     [.0, .0, .0, .9, .0],
                     [.0, .9, .9, .0, .9],
                     [.9, .0, .9, .9, .0]])
    ############################################
elif Nch>=10:
    rvs  = sp.stats.uniform(loc=0.8, scale=.2).rvs
    S    = sp.sparse.random(1, Nch*(Nch-1), density=0.7,  data_rvs=rvs).toarray()[0]
    K    = np.ones((Nch,Nch))-np.eye(Nch)
    K[K==1] = S
    K    = np.tril(K.T).T + 0.5*np.tril(K)

A        = 3.25 + 0.1 * np.random.randn(Nch)
a        = 100  * np.ones(Nch)
B        = 22   + 0.1 * np.random.randn(Nch)
b        = 50   * np.ones(Nch)
p_m      = 220     
##############################################
res = []

# for tri in range(Ntri):
#     out = generate_eeg(fs, dt, Nch, Nt, T_spurious, K, alp)
    
#     res.append(out)
    
#     print(tri)


res  = Parallel(n_jobs=-1, verbose=8)(delayed(generate_eeg)
                                 (fs, dt, Nch, Nt, T_spurious, K, alp, 
                                  A, a, B, b, p_m)
                                     for tri in range(Ntri))

eeg_true   = [val[0] for val in res]
eeg_obs    = [val[1] for val in res]
param_true = res[0][2]
time       = res[0][3] 
#%%
save_dict = {}
save_dict['eeg_true']   = eeg_true
save_dict['eeg_obs']    = eeg_obs
save_dict['time']       = time
save_dict['param_true'] = param_true
save_dict['K_true']     = K
save_dict['alp']        = alp
save_dict['fs']         = fs
save_dict['dt']         = dt
save_dict['Nch']        = Nch
save_dict['Ntri']       = Ntri
save_dict['T']          = T
save_dict['Ttr']        = Ttr

save_name     = 'synthetic_eeg_Nch%02d'%Nch
fullpath_save = save_dir + save_name 
np.save(fullpath_save, save_dict)