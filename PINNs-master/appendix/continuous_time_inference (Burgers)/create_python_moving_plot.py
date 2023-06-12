#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:05:45 2023

@author: gs1511
"""
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import scipy.io

import sys
sys.path.insert(0, '../../Utilities/')

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
data = scipy.io.loadmat('../Data/burgers_shock.mat')


m = 
X_u_train_x = X_u_train[X_u_train[:,1]==0,:]

fig = 