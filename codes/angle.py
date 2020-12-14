#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 07:20:44 2020

@author: aswinmuralidharan
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from matplotlib.ticker import MaxNLocator
my_locator = MaxNLocator(4)

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def dot(a, b):
  return np.sum(a * b, axis=-1)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    """
    cosTh = np.dot(v1, v2)
    sinTh = np.cross(v1, v2)
    angle = np.arctan2(sinTh,cosTh)
    if angle < 0:
        angle = angle+2*math.pi
    return angle

def anglevec(filename, lagtime = 1, maxtime = 100, mpp=0.16, ensemble=False, bins=24):
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
    """
    # Reindex with consecutive frames, placing NaNs in the gaps.
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']
    angle = []
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
        disp = pos.sub(pos.shift(lagtime)).to_numpy()
        # Let np.histogram choose the best bins for all the data together.
        for row in np.arange(0, np.shape(disp)[0]-1):
            angle.append(angle_between(disp[row,:], disp[row+1,:]))    
    angles = pd.DataFrame(angle).dropna()
    return angles
        

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bpc = ['100bp' , '250bp', '500bp']
bpcs = ['100 bp' , '250 bp', '500 bp']
mkfc = ['#fc8d59','#ffffbf', '#91bfdb']
mkr = ['^' , 'o', 's']
fig1,(ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(3.375*2,3.375*2/3))        
ax = (ax1, ax2, ax3)    
i=0
for bp in bpc:
    bps = bpcs[i]    
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tf'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            angle = anglevec(os.path.join(directory, filename)) 
    x_his = angle.to_numpy()
    counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2.001*math.pi,math.pi/20), density = True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    ax1.plot(bin_centres,counts,marker = mkr[i], markerfacecolor = mkfc[i], markeredgecolor = 'k', linestyle = 'None', label = bps )
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            angle = anglevec(os.path.join(directory, filename),5) 
    x_his = angle.to_numpy()
    counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2.001*math.pi,math.pi/20), density =True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    ax2.plot(bin_centres,counts,marker = mkr[i], markerfacecolor = mkfc[i], markeredgecolor = 'k', linestyle = 'None', label = bps)
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            angle = anglevec(os.path.join(directory, filename),20) 
    x_his = angle.to_numpy()
    counts,bin_edges = np.histogram(x_his,bins = np.arange(0,2.001*math.pi,math.pi/20), density = True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    ax3.plot(bin_centres,counts,marker = mkr[i], markerfacecolor = mkfc[i], markeredgecolor = 'k', linestyle = 'None', label = bps)
    i+=1 

ax1.legend(loc ='upper left', frameon = False,  handletextpad=0.1)       
for n, ax in enumerate((ax1,ax2,ax3)):   
    ax.text(-0.2, 1, r'\textbf{'+ string.ascii_lowercase[n]+'}', transform=ax.transAxes, 
            size=8, weight='bold')             
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis="both", direction="in")
    ax.set_ylim(0,0.8)
    ax.set_xlim(0,2*math.pi)
    x_pi   = bin_centres/np.pi
    unit   = 0.5
    x_tick = np.arange(0, 2+unit, unit)
    x_label = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$",   r"$2\pi$"]
    ax.set_xticks(x_tick*np.pi)
    ax.set_xticklabels(x_label)
    ax.set(xlabel=r'$\theta$ (rad) ',
            ylabel=r'$P(\theta, \Updelta t)$ ')
    ax.yaxis.set_major_locator(my_locator)
ax1.text(0.95, 0.9, r'$\Updelta t = 0.1$ s', verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax1.transAxes)
ax2.text(0.95, 0.9, r'$\Updelta t = 0.5$ s', verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax2.transAxes)
ax3.text(0.95, 0.9, r'$\Updelta t = 1$ s', verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax3.transAxes)
plt.tight_layout()
fig1.savefig(directory3 + '/angle.pdf')