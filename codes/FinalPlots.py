#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:40:46 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
markers = ['rs', 'go', 'b^', 'kD']
bp = ['100bp' , '250bp', '500bp', '1000bp']  # Base pair to process
bps = ['100 bp', '250 bp', '500 bp', '1000 bp']
fig1, ax1 = plt.subplots()
for i in np.arange(4):
    directory = Filepath + '/E_output_data/' + bp[i] + '/tm'
    filename = Filepath + '/E_output_data/' + bp[i] + '/MSDcollected/' +  'MSDcollected.csv'
    MSDcollected_df = pd.read_csv(filename).set_index('lag time [s]')
    ax1.plot(MSDcollected_df.index, MSDcollected_df.mean(1), markers[i], label = bps[i])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax1.set_ylim((1e-3, 0.2))
ax1.set_xlim((0.07,15 ))
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which="both", axis="both", direction="in")
ax1.set_aspect(1, adjustable='box')
ax1.legend(frameon = False, handletextpad=0.1)