#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:20:09 2023

@author: gs1511
"""
import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
#from plotting import newfig, savefig

import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec


t = np.genfromtxt('t.csv',delimiter=',')
x = np.genfromtxt('x.csv',delimiter=',')
Exact  = np.genfromtxt('Exact.csv',delimiter=',')
f_pred  = np.genfromtxt('f_pred.csv',delimiter=',')
u_pred  = np.genfromtxt('u_pred.csv',delimiter=',')
u_train  = np.genfromtxt('u_train.csv',delimiter=',')
U_pred  = np.genfromtxt('U_pred.csv',delimiter=',')
idx  = np.genfromtxt('idx.csv',delimiter=',')

X_u_train =np.genfromtxt('X_u_train.csv',delimiter=',')
X_f_train =np.genfromtxt('X_f_train.csv',delimiter=',')

######################################################################
############################# Plotting ###############################
######################################################################    


#fig, ax = newfig(1.0, 1.1)


fig = plt.figure()

gs = GridSpec(nrows= 2, ncols=3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
####### Row 0: u(t,x) ##################    
gs0 = GridSpec(1, 3)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax1 = plt.subplot(gs0[:, :])

h = ax1.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax1.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax1.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
 
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend(frameon=False, loc = 'best')
ax1.set_title('u(t,x)', fontsize = 10)


####### Row 1: u(t,x) slices ##################    
gs1 = GridSpec(1, 3)
gs1.update(top=1-1/2, bottom=0, left=0.1, right=0.9, wspace=0.3)

ax = plt.subplot(gs1[0:1, 0])
ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('x')
ax.set_ylabel('xu(t,x)')    
ax.set_title('t = 0.25', fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0:1, 1])
ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('x')
#ax.set_ylabel('u(t,x)')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('t = 0.50', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0:1, 2])
ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('x')
#ax.set_ylabel('u(t,x)')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('t = 0.75', fontsize = 10)
ax.set_aspect('equal')
#ig.subplots_adjust(wspace = 0,hspace=0)
#plt.subplots_adjust(wspace=None)
#plt.axes().set_aspect('equal')
plt.show()


fig1 = plt.figure(constrained_layout=True)
spec1=  fig1.add_gridspec(ncols = 3, nrows=2)
plt.show()