#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:49:45 2020

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

def power_law(x,alpha, A):
    return A*np.power(x, alpha)


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


# Filepath and files to process

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
#bps = ['100bp', '250bp','500bp']
#bpcs = ['100 bp', '250 bp','500 bp']
bps = ['MCF7500bp']
bpcs = ['MCF7 500bp']
mpp = 0.16
fps = 10
max_lagtime = 100
pos_columns = ['x', 'y']
i = 0
num_plots = 1
axnum=0

# Pre-initialize the plots
fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(3.375,3.375*2*1.6/3))
fig2, ((ax5),(ax6),(ax7)) = plt.subplots(3,1,figsize=(3.375,3.375*0.4*3),sharex=True)
ax = (ax1,ax2,ax3)
ax2 = (ax5, ax6, ax7)
colors = ['#fc8d59','#ffffbf', '#91bfdb']
#colors = ['#FFFFFF','#FFFFFF','#FFFFFF']
mkr = ['^' , 'o', 's']
delta = 1
z= [2,1,0] 


for bp in bps:      
    j=0
    video = 1
    dispfromoriginmax = []
    trajec = []
    corr = []
    corr0 = []
    disp = []
    disp2 = []
    MSDt = []
    alpha = []
    ax[i].axhline(y=0, color='k', linestyle='dashdot')
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
        x = traj['x']
        y = traj['y']
        disp.append(np.sqrt(x.sub(x.shift(delta))**2 + y.sub(y.shift(delta))**2))
        disp2.append(x.sub(x.shift(delta))**2 + y.sub(y.shift(delta))**2)
        disp2temp = []
        for tau in np.arange(0,100):
            disp2temp.append((x.sub(x.shift(delta))**2 + y.sub(y.shift(delta))**2).head(tau).mean())
        MSDt.append(disp2temp)
        tempt = np.arange(0,100)/10
        disp2temp = np.array(disp2temp)
        tempt = tempt[np.logical_not(np.isnan(disp2temp))]
        disp2temp = disp2temp[np.logical_not(np.isnan(disp2temp))]
        if len(disp2temp)>50:
            pars, cov = curve_fit(f = power_law, xdata = tempt[8:], ydata = disp2temp[8:], p0=[0, 0], bounds=(-np.inf, np.inf))
            alpha.append(pars)
    MSDt = pd.DataFrame(MSDt).T
    t = MSDt.index.values.astype('float64') / float(10)
    MSDt.set_index(t, inplace=True)
    MSDt.index.name = 'time [s]'
    MSDav = MSDt.mean(axis = 1)
    msdnp = np.array(MSDav)
    tempt = np.arange(0,100)/10
    pars2, cov2 = curve_fit(f = power_law, xdata = tempt[8:], ydata = msdnp[8:], p0=[0, 0], bounds=(-np.inf, np.inf))
    print(pars2)
    alpha = np.array(alpha)[:,0]
    ax[i].plot(MSDt, color = 'lightgray')
    ax[i].plot(MSDav , marker = 'o' , markeredgecolor = 'black' , markerfacecolor = '#228a8d', linestyle = 'None')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set(ylabel=r'$\langle \overline {\Updelta \mathbf{r}^2(\Updelta t, T)}\rangle$ ($\mathrm{\upmu}$m$^2$)',
            xlabel=r'$T$ (s)')
    ax[i].set_ylim((1e-6, 1))
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(which="both", axis="both", direction="in")
    ax[i].xaxis.set_major_locator(LogLocator(base = 10, numticks =5))
    locmin = LogLocator(base=10.0,subs=tuple(np.arange(0.1, 1, 0.1)),numticks=5)
    ax[i].xaxis.set_minor_locator(locmin)
    ax[i].xaxis.set_minor_formatter(NullFormatter())
    
    ax[i].yaxis.set_major_locator(LogLocator(base = 10, numticks =7))
    locmin = LogLocator(base=10.0,subs=tuple(np.arange(0.1, 1, 0.1)),numticks=7)
    ax[i].yaxis.set_minor_locator(locmin)
    ax[i].yaxis.set_minor_formatter(NullFormatter())
    
    ax[i].text(0.9, 0.92, bpcs[i] + '\n' + r'$\Updelta t$  = 0.1 s' , verticalalignment='top', horizontalalignment='right',
             multialignment="left",
             transform=ax[i].transAxes)
    ax[i].text(-0.4, 1, r'\textbf{'+ string.ascii_uppercase[i]+'}', transform=ax[i].transAxes, 
            size=8, weight='bold')
    ax2[i].hist(alpha,bins = np.arange(-1.05,1.05,0.1), edgecolor='black',color = 'white', density = True, rwidth=1)
    ax2[i].set(ylabel=r'$P(\beta)$')
    ax2[i].yaxis.set_ticks_position('both')
    ax2[i].xaxis.set_ticks_position('both')
    ax2[i].tick_params(which='both', axis="both", direction="in")
    ax2[i].set_xlim((-1, 1))
    ax2[i].set_ylim((0, 2.5))
    ax2[i].text(0.95, 0.9, bpcs[i], verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax2[i].transAxes, color = 'black')
    ax2[i].text(-0.12, 1, r'\textbf{'+ string.ascii_uppercase[i]+'}', transform=ax2[i].transAxes, 
            size=8, weight='bold')
    EB1 = np.square(MSDt).mean(axis = 1)
    EB2 = np.square(MSDav)
    EB = (EB1-EB2)/EB2
    ax4.plot(EB[::2], marker = mkr[i], markeredgecolor = 'black',  markerfacecolor = colors[i], linestyle = 'None', label = bpcs[i], zorder = z[i])
    i+=1
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_ylim((1e-1, 10))
ax4.set(ylabel=r'$EB$',
            xlabel=r'$T$ (s)')
ax4.yaxis.set_ticks_position('both')
ax4.xaxis.set_ticks_position('both')
ax4.tick_params(which="both", axis="both", direction="in")
ax4.text(-0.4, 1, r'\textbf{'+ string.ascii_uppercase[3]+'}', transform=ax4.transAxes, 
            size=8, weight='bold')
ax4.legend(loc ='lower left', frameon = False,  handletextpad=0.1)
ax4.xaxis.set_major_locator(LogLocator(base = 10, numticks =5))
locmin = LogLocator(base=10.0,subs=tuple(np.arange(0.1, 1, 0.1)),numticks=5)
ax4.xaxis.set_minor_locator(locmin)
ax4.xaxis.set_minor_formatter(NullFormatter())
ax2[2].set(xlabel=r'$\beta$')
fig1.tight_layout()  
fig2.tight_layout()   

fig1.savefig(directory3 + '/Ergodic.pdf')  
fig2.savefig(directory3 + '/aging.pdf')