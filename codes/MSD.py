#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:39:14 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection

def _msd_iter(pos, lagtimes):   
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        yield np.concatenate((np.nanmean(diff, axis=0),np.nanmean(diff**2, axis=0)))
        
plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')
pos_columns = ['x', 'y']
result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]
                     
df = pd.ExcelFile('/Users/aswinmuralidharan/OneDrive/Paper2/NARtemplatev1/Data/100bp.xlsx').parse('Sheet1').to_numpy() #you could add index_col=0 if there's an index
fig, ax = plt.subplots()
t = np.arange(0,10.1,0.1)
max_lagtime = 100
MSD = np.zeros((max_lagtime-1,256))
for i in np.arange(0,452,2):
    r = df[1:,i:i+2]
    max_lagtime = min(max_lagtime, len(r) - 1)  # checking to be safe
    lagtimes = np.arange(1, max_lagtime + 1)
    result = pd.DataFrame(_msd_iter(r, lagtimes),columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    MSD[:,int(i/2)] = result['msd'].to_numpy()
    plt.loglog(lagtimes/10, result['msd'].to_numpy(),'k',alpha=0.5)
    ax.set(ylabel=r'$\langle \Delta r^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
       xlabel=r'$\mathrm{\tau}$ (s)')
    ax.tick_params(which="both", axis="both", direction="in")