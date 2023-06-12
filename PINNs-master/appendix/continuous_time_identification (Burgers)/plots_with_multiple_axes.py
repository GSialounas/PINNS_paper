#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:18:05 2023

@author: gs1511
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(.5, .5, "ax%d" % (i+1), va = "center", ha = "center")
        ax.tick_params(axis = 'both',labelbottom=False, labelleft=False)
        #print("ax%d"%(i+1))
        #ax.tick_params(labelbottom = False, labeleft=False)
        
fig = plt.figure(layout="constrained")
gs = GridSpec(3,3, figure = fig)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[1,:-1])
ax2.text(.5, .5, "ax%d" % (2), va = "center", ha = "center")
ax3 = fig.add_subplot(gs[1:,-1])
ax4 = fig.add_subplot(gs[-1,0])
ax5 = fig.add_subplot(gs[-1,-2])
fig.suptitle("Gridpsec")
format_axes(fig)
plt.show()
