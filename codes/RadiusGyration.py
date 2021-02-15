#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:04:16 2021

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from scipy.optimize import curve_fit
from matplotlib.ticker import LogLocator, NullFormatter

plt.style.use('aswinplotstyle')

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
plt.rc('font', size=7)
plt.rc('axes', titlesize=7)
plt.rc('axes', labelsize=7)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
       r"\usepackage{amsmath}"
       r"\usepackage{textgreek}"
       r"\usepackage{upgreek}"
]  
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif'
plt.rcParams['axes.linewidth'] = 1


def trajectory(pos, pid, max_lagtime,video, pos_columns):
    """Shift the starting point of the trajectory to (0,0).
    Parameters
    ----------
    pos : Positions x, and y for individual particles 
    max_lagtime : intervals of frames out to which MSD is computed
    Returns
    -------
    DataFrame([<x>, <y>, pid], index=t)
    Notes
    -----
    Input units are microns and frames. Output units are microns and frames."""

    max_lagtime = len(pos)
    lagtimes = np.arange(0, max_lagtime)
    result = pd.DataFrame(pos - pos[0], columns=pos_columns, index=lagtimes)
    result['particle'] = pid
    result['video'] = video
    return result

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
#bps = ['100bp', '250bp','500bp']
#bpcs = ['100 bp', '250 bp','500 bp']
bps = ['MCF7500bp']
bpcs = ['MCF7 500bp']

mpp = 0.16
fps = 10
max_lagtime = 100
pos_columns = ['x', 'y']
trajec = []
fig1 = plt.figure(figsize=(3.375*2,3.375*2*1.6/3))

ax1 = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan =1)
ax2 = plt.subplot2grid((2, 3), (0, 0) )
ax3 = plt.subplot2grid((2, 3), (1, 0))
ax4 = plt.subplot2grid((2, 3), (1, 1))
ax5 = plt.subplot2grid((2, 3), (1, 2))
axs = (ax3,ax4,ax5)

bps = ['100bp', '250bp','500bp']
bpcs = ['100 bp', '250 bp','500 bp']
rg = []
video = 1
axnum = 0
for bp in bps:
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tf'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures' 
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            traj = pd.read_csv(os.path.join(directory, filename))
            for pid, ptraj in traj.groupby('particle'):
                pos = ptraj.set_index('frame')[pos_columns] * mpp
                pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
                trajec.append(trajectory(pos, pid, max_lagtime,video, pos_columns))
                resultstraj = pd.concat(trajec)
            time = resultstraj.index.values.astype('float64') / float(fps)
            resultstraj.set_index(time, inplace=True)
            resultstraj.index.name = 'Time [s]' 
            video +=1
    for id, traj in resultstraj.groupby(['particle', 'video']):
        x = traj['x'].head(200)
        y = traj['y'].head(200)
        xcom = np.nanmean(x)
        ycom = np.nanmean(y)
        xsqdis = [(x1 - xcom)**2 for x1 in x]
        ysqdis = [(y1 - ycom)**2 for y1 in y]
        xrg = np.sqrt(np.nanmean(xsqdis))
        yrg = np.sqrt(np.nanmean(ysqdis))
        rg.append(np.sqrt(xrg**2 + yrg**2))    
    axs[axnum].hist(rg,bins = np.arange(0,1.01,0.05), edgecolor='black',color = 'white', density = True, rwidth=1)
    axs[axnum].yaxis.set_ticks_position('both')
    axs[axnum].xaxis.set_ticks_position('both')
    axs[axnum].tick_params(which='both', axis="both", direction="in")
    axs[axnum].set_xlim((-0.05, 1.05))
    axs[axnum].set_ylim((0, 10))
    axs[axnum].text(0.95, 0.95, bpcs[axnum], verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=axs[axnum].transAxes, color = 'black')
    axs[axnum].set(xlabel=r'$ \left |\mathbf{R}_{\mathrm{g}}\right |$' +' '+ r'(\textmu m) ',
        ylabel=r'$P(\left |\mathbf{R}_{\mathrm{g}}\right |)$'+' '+ r'(\textmu m$^{-1}$) ')
    axnum += 1
for n, ax in enumerate((ax3, ax4, ax5)):   
    ax.text(-0.2, 1, r'\textbf{'+ string.ascii_uppercase[n+2]+'}', transform=ax.transAxes, 
            size=8)
ax1.text(-0.2, 1, r'\textbf{'+ string.ascii_uppercase[1]+'}', transform=ax1.transAxes, 
        size=8)
ax2.text(-0.2, 1, r'\textbf{'+ string.ascii_uppercase[0]+'}', transform=ax2.transAxes, 
        size=8)
ax1.axis("off")
ax2.axis("off")
fig1.tight_layout()

fig1.savefig(directory3 + '/Rg.png', dpi = 300)