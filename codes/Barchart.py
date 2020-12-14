#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:48:20 2020

@author: aswinmuralidharan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import add, truediv
import os
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 2.0  # previous svg hatch linewidth
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
bp = ['100bp' , '250bp', '500bp']
bps = ['100 bp', '250 bp', '500 bp']
if not os.path.exists(Filepath + '/E_output_data/Combined/Figures'):
    os.makedirs(Filepath + '/E_output_data/Combined/Figures')
fig1, ax1 = plt.subplots()
MSDsublen = []
MSDsuperlen = []
directory3 = Filepath + '/E_output_data/Combined' + '/Figures'
N = 3
for i in np.arange(3):
    directory = Filepath + '/E_output_data/' + bp[i] + '/tm'
    filename = Filepath + '/E_output_data/' + bp[i] + '/MSDcollected/' +  'MSDsub.csv'
    MSDsub_df = pd.read_csv(filename).set_index('lag time [s]')
    filename = Filepath + '/E_output_data/' + bp[i] + '/MSDcollected/' +  'MSDsuper.csv'
    MSDsuper_df = pd.read_csv(filename).set_index('lag time [s]')
    MSDsublen.append(len(MSDsub_df.columns))
    MSDsuperlen.append(len(MSDsuper_df.columns))
Total = list(map(add, MSDsublen, MSDsuperlen) ) 
MSDsup = list(map(truediv, MSDsuperlen, Total))
MSDsup = list(map(lambda x: x * 100, MSDsup)) 
MSDsub = list(map(truediv, MSDsublen, Total)) 
MSDsub = list(map(lambda x: x * 100, MSDsub))  
ind = np.arange(N)    # the x locations for the groups
width = 0.5       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, MSDsub, width, fill=False, edgecolor='black', hatch="/", linewidth=1, label = 'Subdiffusive'+'\n'+r'$0<\alpha_2<1$', alpha=1)
p2 = plt.bar(ind, MSDsup, width,fill=False, bottom = MSDsub, edgecolor='black', linewidth=1, hatch='..', label = 'Superdiffusive'+'\n'+r'$1<\alpha_2<2$', alpha=1)
ax1.tick_params(which="both", axis="both", direction="in")
ax1.set_ylabel('Percentage')
plt.xticks(ind, ('100 bp', '250 bp', '500 bp'))
ax1.set_ylim(0,120)
ax1.set_aspect(0.025, adjustable='box')
ax1.legend(loc='center left',frameon= False, bbox_to_anchor=(1, 0.5))
plt.tight_layout()
fig1.savefig(directory3 + '/barlate.png', dpi = 300)