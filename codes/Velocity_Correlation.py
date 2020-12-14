#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:01:53 2020

@author: aswinmuralidharan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string

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
bps = ['100bp', '250bp','500bp']
bpcs = ['100 bp', '250 bp','500 bp']
mpp = 0.16
fps = 10
max_lagtime = 100
pos_columns = ['x', 'y']
i = 0
num_plots = 1
axnum=0
fig1, (ax1) = plt.subplots(1,1, figsize=(3.375,0.6*3.375))
mkfc = ['#fc8d59','#ffffbf', '#91bfdb']
mkr = ['^' , 'o', 's']
for bp in bps:      
    j=0
    for tau in [2]:
        corrv = []
        xdisp0 = []
        ydisp0 = []
        video = 1
        dispfromoriginmax = []
        trajec = []
        corr = []
        corr0 = []
        ax1.axhline(y=0, color='k', linestyle='dashdot')
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
        for delta in np.arange(0,100):       
            xdisp = []
            ydisp = []
            for id, traj in resultstraj.groupby(['particle', 'video']):
                x = traj['x']
                xdisp.append(x.sub(x.shift(tau)))
                y = traj['y']
                ydisp.append(y.sub(y.shift(tau)))
            xdisp = pd.concat(xdisp, axis =1)
            ydisp = pd.concat(ydisp, axis =1)
            corr.append((xdisp.mul(xdisp.shift(delta)).mean().mean()+ydisp.mul(ydisp.shift(delta)).mean().mean())/(xdisp.mul(xdisp.shift(0)).mean().mean()+ydisp.mul(ydisp.shift(0)).mean().mean()))
        ax1.plot(np.arange(0,10,0.1)/(tau/10),corr,linestyle = 'None', marker = mkr[i], markerfacecolor = mkfc[i], markeredgecolor = 'k', label = bp)
        j+=1
    i+=1
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which='both', axis="both", direction="in")
ax1.set(xlabel=r'$\Updelta t/\delta$',
ylabel=r'$\mathrm{C}_\mathrm{v}^{\delta}(\Updelta t)/\mathrm{C}_\mathrm{v}^{\delta}(\Updelta t = 0)$')
ax1.set_xlim(0, 5)
ax1.set_ylim(-0.5,1)
ax1.set_yticks([-0.5, 0 , 0.5, 1])
ax1.legend(loc ='upper right', frameon = False,  handletextpad=0.1)
plt.tight_layout()
fig1.savefig(directory3 + '/corr.pdf', dpi = 300)