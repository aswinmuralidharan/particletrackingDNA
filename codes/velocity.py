#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:38:41 2020

@author: aswinmuralidharan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:07:23 2020

@author: aswinmuralidharan
"""
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')


def velocity(filename, lagtime = 1, maxtime = 100, mpp=0.16, ensemble=False, bins=24):
    """Compute the van Hove correlation (histogram of displacements).
    The van Hove correlation function is simply a histogram of particle
    displacements. It is useful for detecting physical heterogeneity
    (or tracking errors).
    Parameters
    ----------
    pos : DataFrame
        x or (or!) y positions, one column per particle, indexed by frame
    lagtime : integer interval of frames
        Compare the correlation function at this lagtime.
    mpp : microns per pixel, DEFAULT TO 1 because it is usually fine to use
        pixels for this analysis
    ensemble : boolean, defaults False
    bins : integer or sequence
        Specify a number of equally spaced bins, or explicitly specifiy a
        sequence of bin edges. See np.histogram docs.
    Returns
    -------
    vh : DataFrame or Series
        If ensemble=True, a DataFrame with each particle's van Hove correlation
        function, indexed by displacement. If ensemble=False, a Series with
        the van Hove correlation function of the whole ensemble.
    Examples
    --------
    >>> pos = traj.set_index(['frame', 'particle'])['x'].unstack() # particles as columns
    >>> vh = vanhove(pos)
    """
    # Reindex with consecutive frames, placing NaNs in the gaps.
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']
    vel = []
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
        disp = pos.sub(pos.shift(lagtime))
        disp = (disp['x']**2+disp['y']**2)**0.5/0.1
        # Let np.histogram choose the best bins for all the data together.
        values = disp.values
        values = values[np.isfinite(values)]
        vel.append(disp.head(maxtime))
    vel = pd.concat(vel, ignore_index= True).dropna()
    return vel
        
Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = '100bp'  # Base pair to process
bps = '100 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
for filename in os.listdir(directory):
    if not filename.startswith('.'):
        velo = velocity(os.path.join(directory, filename)) 
fig1,ax1 = plt.subplots(figsize=(5,5))
x_his = velo.to_numpy()/0.001
#weights = np.ones_like(x_his)/len(x_his)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2200,200), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
ax1.plot(bin_centres,counts,marker = 'o', markerfacecolor = '#fc8d59', markeredgecolor = 'k', linestyle = 'None', label = '500 bp' )

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = '250bp'  # Base pair to process
bps = '250 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
for filename in os.listdir(directory):
    if not filename.startswith('.'):
        velo = velocity(os.path.join(directory, filename)) 
x_his = velo.to_numpy()/0.001
#weights = np.ones_like(x_his)/len(x_his)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2200,200), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
ax1.plot(bin_centres,counts,marker = 'o', markerfacecolor = '#ffffbf', markeredgecolor = 'k', linestyle = 'None', label = '500 bp' )

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = '500bp'  # Base pair to process
bps = '500 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
for filename in os.listdir(directory):
    if not filename.startswith('.'):
        velo = velocity(os.path.join(directory, filename)) 
x_his = velo.to_numpy()/0.001
#weights = np.ones_like(x_his)/len(x_his)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2200,200), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
ax1.plot(bin_centres,counts,marker = 'o', markerfacecolor = '#91bfdb', markeredgecolor = 'k', linestyle = 'None', label = '500 bp' )










ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which='both', axis="both", direction="in")
ax1.set_yscale('log')
#ax1.set_ylim(0.01,1)
ax1.set_xlim(0,2500)
ax1.set(xlabel=r'$\mid v \mid$ (nm/s) ',
        ylabel=r'$P(\mid v \mid)$ ')
ax1.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax1.transAxes)
#ax1.yaxis.set_ticks(np.arange(0, 0.2, step=0.05))











fig1.savefig(directory3 + '/velocity.eps')