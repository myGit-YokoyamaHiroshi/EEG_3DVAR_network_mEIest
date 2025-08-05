# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:41:34 2024

@author: H.Yokoyama
"""
from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np

#%%
##############################################################################

def Sigm(v):
    v0   = 6
    vmax = 5
    r    = 0.56
    sigm = vmax / (1 + np.exp(r * ( v0 - v )))
    
    return sigm

def postsynaptic_potential_function(y, z, A, a, Sgm):
    dy = z
    dz = A * a * Sgm - 2 * a * z - a**2 * y
    
    f_out = np.vstack((dy, dz))
    return f_out

def func_JR_network_model(y, *args):#A, a, B, b, u, K, alp):
    A   = args[0]
    a   = args[1]
    B   = args[2]
    b   = args[3]
    u   = args[4]
    K   = args[5]
    alp = args[6]
    
    Nvar, Nch = y.shape
    
    dy  = np.zeros(y.shape)

    ### column 1 ##############################################################
    ### Connectivity constants ######
    C1   = 135 # num. of neurons
    c1   = 1.0  * C1
    c2   = 0.8  * C1
    c3   = 0.25 * C1
    c4   = 0.25 * C1
    #################################
    
    f        = Sigm(y[1,:] - y[2,:])
    sum_term = K @ f[:,np.newaxis]
    
    Sgm_12 = Sigm(y[1,:] - y[2,:]);
    Sgm_p0 = u + c2 * Sigm(c1*y[0,:]) + alp * sum_term.reshape(-1);
    Sgm_0  = c4 * Sigm(c3*y[0,:]);
        
    dy_03 = postsynaptic_potential_function(y[0, :], y[3, :], A, a, Sgm_12);
    dy_14 = postsynaptic_potential_function(y[1, :], y[4, :], A, a, Sgm_p0);
    dy_25 = postsynaptic_potential_function(y[2, :], y[5, :], B, b, Sgm_0);
    
    # sort order of dy1
    dy[0,:] = dy_03[0,:]
    dy[3,:] = dy_03[1,:]
    
    dy[1,:] = dy_14[0,:]
    dy[4,:] = dy_14[1,:]
    
    dy[2,:] = dy_25[0,:]
    dy[5,:] = dy_25[1,:]
    ##########################################################################
 
    return dy


def runge_kutta(dt, func, X_now, *args):
    k1   = func(X_now, *args)
    
    X_k2 = X_now + (dt/2)*k1
    k2   = func(X_k2, *args)
    
    X_k3 = X_now + (dt/2)*k2
    k3   = func(X_k3, *args)
    
    X_k4 = X_now + dt*k3
    k4   = func(X_k4, *args)
    
    X_next = X_now + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return X_next


