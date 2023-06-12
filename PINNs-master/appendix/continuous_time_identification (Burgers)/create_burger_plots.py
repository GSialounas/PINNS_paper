#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:45:58 2023

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



#import time

t = np.genfromtxt('t.csv',delimiter=',')
x = np.genfromtxt('x.csv',delimiter=',')
Exact  = np.genfromtxt('Exact.csv',delimiter=',')
f_pred  = np.genfromtxt('f_pred.csv',delimiter=',')
u_pred  = np.genfromtxt('u_pred.csv',delimiter=',')
X_u_train =np.genfromtxt('X_u_train.csv',delimiter=',')
u_train =np.genfromtxt('u_train.csv',delimiter=',')

lambda_1_value =np.genfromtxt('lambda_1_value.csv',delimiter='')
lambda_2_value =np.genfromtxt('lambda_2_value.csv',delimiter='')
error_u =np.genfromtxt('error_u.csv',delimiter='')

N_u = 2000
X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              
idx = np.random.choice(X_star.shape[0], N_u, replace=False)

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)  

x1 = np.linspace(-1,1,50)
x2 = np.linspace(-2,2,100)
x3 = np.linspace(-3,3,150)
x4 = np.linspace(-1,1,50)
x5 = np.linspace(-2,2,100)
x6 = np.linspace(-3,3,150)




def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

fig = plt.figure(layout="constrained")

gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])

####### Row 0: u(t,x) ##################    
gs0 = GridSpec(1, 3)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax1 = plt.subplot(gs0[:, :])

h = ax1.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)



ax1.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax1.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
ax1.set_title('u(t,x)', fontsize = 10)

####### Row 1: u(t,x) slices ##################    
#gs1 = GridSpec(1, 3)
#gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax2 = fig.add_subplot(gs[1, 0])

#ax2 = plt.subplot(gs[1, 0])
ax2.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax2.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax2.set_xlabel('x')
ax2.set_ylabel('u(t,x)')    
ax2.set_title('t = 0.25', fontsize = 10)
ax2.axis('square')
ax2.set_xlim([-1.1,1.1])
ax2.set_ylim([-1.1,1.1])

# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax3.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax3.set_xlabel('x')
ax3.set_ylabel('u(t,x)')    
ax3.set_title('t = 0.25', fontsize = 10)
ax3.axis('square')
ax3.set_xlim([-1.1,1.1])
ax3.set_ylim([-1.1,1.1])
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax4.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax4.set_xlabel('x')
ax4.set_ylabel('u(t,x)')    
ax4.set_title('t = 0.25', fontsize = 10)
ax4.axis('square')
ax4.set_xlim([-1.1,1.1])
ax4.set_ylim([-1.1,1.1])
#ax5 = fig.add_subplot(gs[-1, -2])

fig.suptitle("PINNS Plot")
#fig.tight_layout()
#format_axes(fig)
plt.savefig("fig_out.png",bbox_inches = 'tight')

plt.show()

"""  
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[1,2])

"""
"""

h = ax1.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
              extent=[t.min(), t.max(), x.min(), x.max()], 
              origin='lower', aspect='auto')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size = "5%", pad = 0.05)
ax1.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax1.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax1.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.legend(loc='upper center', bbox_to_anchor=(.85, -0.125), ncol=5, frameon=False)
ax1.set_title('u(t,x)', fontsize = 10)


#ax1.plot(x,np.sin(5*np.pi*x))
#ax1.plot(plot_dict["xy0"][0], plot_dict["xy0"][1])



##ax3.plot(x,.5*x)
#ax4.plot(x,1.5*x)
#ax5.plot(plot_dict["xy0"][0], plot_dict["xy0"][1])

fig.suptitle("Gridpsec")
#format_axes(fig,plot_dict)

plt.show()

"""