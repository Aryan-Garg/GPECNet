# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 22:30:49 2021

@author: hp
"""
import numpy as np
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
        
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

p = frange_cycle_linear(650)
print(p)