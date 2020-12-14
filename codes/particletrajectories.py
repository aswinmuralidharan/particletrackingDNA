#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:09:04 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')

df = pd.ExcelFile('/Users/aswinmuralidharan/OneDrive/Paper2/NARtemplatev1/Data/100bp.xlsx').parse('Sheet1').to_numpy() #you could add index_col=0 if there's an index
fig = plt.figure()
axs = fig.add_subplot(111)
axs.axhline(y=0, color='k', linestyle='dashdot')
axs.axvline(x=0, color='k', linestyle='dashdot')
t = np.arange(0,10.1,0.1)
for i in np.arange(0,478,2):
    x = df[2:,i]
    y = df[2:,i+1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    del x,y
fig.colorbar(line, ax=axs).ax.tick_params(axis='y', direction='in')
plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)
axs.set(ylabel=r'y ($\mathrm{\mu}$m)',
        xlabel=r'x ($\mathrm{\mu}$m)')
axs.set_aspect('equal', adjustable='box')
plt.minorticks_on()
axs.yaxis.set_ticks_position('both')
axs.xaxis.set_ticks_position('both')
axs.tick_params(which='both', axis="both", direction="in")
plt.savefig('/Users/aswinmuralidharan/OneDrive/Paper2/NARtemplatev1/Data/100bp.pdf')