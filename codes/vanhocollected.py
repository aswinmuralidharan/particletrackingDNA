#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:12:52 2020

@author: aswinmuralidharan
"""
import pandas as pd
from scipy.optimize import curve_fit
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

def fitGaussian(x,a,sigma):
    return (a*np.exp(-((x)**2/(2*sigma))))

def fitexp(x,a,sigma):
    return (a*np.exp(-((x)/(2*sigma))))

def vanhove(filename, lagtime = 1, maxtime = 100, mpp=0.16):
    """
    Return particle displacements for specified lag time without any particle tags
    """
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']
    vanho = []
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
        disp = pos.sub(pos.shift(lagtime))
        # Let np.histogram choose the best bins for all the data together.
        values = disp.values
        values = values[np.isfinite(values)]
        vanho.append(disp.head(maxtime))
    vanho = pd.concat(vanho, ignore_index= True).dropna()
    return vanho

def kurtosis(data):
    """
    Return excess kurtosis of the data
    """
    x_4 = np.power(data,4)
    x_2 = np.power(data,2)
    kurtosis = np.mean(x_4)/(np.mean(x_2)**2)-3
    return kurtosis
"""
Create figure with required size and arrangement
"""
fig1,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(3.375,3.375))
bpc = ['100bp' , '250bp', '500bp']
bpcs = ['100 bp' , '250 bp', '500 bp']
mkfc = ['#fc8d59','#ffffbf', '#91bfdb']
mkr = ['^' , 'o', 's']
ax = (ax1, ax2, ax3, ax4)       
Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
i = 0
"""
Looping over files

"""
for bp in bpc:
    """
    Checking for directories
    
    """    
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tf'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
    """
    Displacement histogram
    """
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            vanho = vanhove(os.path.join(directory, filename)) 
    bps = bpcs[i]    
    x =  vanho['x'].append(vanho['y']).reset_index()       
    x_his = np.abs(x.to_numpy()/0.001)
    counts,bin_edges = np.histogram(x_his,bins = np.arange(0,300,20), density = True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    bin_centres2 = bin_centres[np.where(bin_centres<=100)]
    """
    Fitting Gaussian part
    """
    counts2 = counts[np.where(bin_centres<=100)]
    bin_centres3 = np.arange(0,200)
    popt, pcov = curve_fit(fitGaussian, bin_centres2, counts2, [0.1,30])
    """
    Fitting exponential part
    """
    bin_centres4 = bin_centres[np.where(bin_centres >=100)]
    counts4 = counts[np.where((bin_centres >=100))]
    popt2, pcov2 = curve_fit(fitexp, bin_centres4, counts4, [0.1,30])
    bin_centres5 = np.arange(0,250)
    
    """
    Plotting
    """
    ax[i].plot(bin_centres3, fitGaussian(bin_centres3,*popt), 'k', linewidth=1, label = 'Gaussian fit')
    ax[i].plot(bin_centres5, fitexp(bin_centres5,*popt2), 'k-.', linewidth=1, label = 'Gaussian fit')
    ax[i].plot(bin_centres,counts, marker = 'o', markerfacecolor = 'lightgray', markeredgecolor = 'k', linestyle = 'None', label = bps)
    """
    Plotstyle
    """
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(which='both', axis="both", direction="in")
    ax[i].set_yscale('log')
    ax[i].set_ylim(1e-5,0.1)
    ax[i].set_xlim(0,250)
    ax[i].set(xlabel=r'$\left |\Updelta x\right|$ (nm) ',
            ylabel=r'$P(\left |\Updelta x\right|, \Updelta t)$ (nm$^{-1}$) ')
    ax[i].text(0.9, 0.92, bps + '\n'+ r'$\Updelta t = 0.1$ s',  verticalalignment='top', horizontalalignment='right',
             multialignment="left",
             transform=ax[i].transAxes)
    """
    Evaluate kurtosis by looping over different lag times
    """
    kurt=[]
    for tau in np.arange(1,51):
        for filename in os.listdir(directory):
            if not filename.startswith('.'):
                vanho = vanhove(os.path.join(directory, filename),tau)         
        x =  vanho['x'].append(vanho['y']).reset_index()       
        x_his = np.abs(x.to_numpy()/0.001)
        kurt.append(kurtosis(x_his))
    """
    Plotting kurtosis
    """
    ax4.plot(np.arange(1,51)/10, kurt, marker = mkr[i], markerfacecolor = mkfc[i], markeredgecolor = 'k', linestyle = 'None', label = bps)
    i+=1
"""
Label expressions on the plot in ax3
"""
ax1.text(0.05, 0.15, r'$e^{-\left|\Updelta x\right|^2/\sigma}$',  verticalalignment='top', horizontalalignment='left',
         multialignment="left",
         transform=ax1.transAxes)
ax1.text(0.95, 0.65, r'$e^{-\left|\Updelta x\right|/\sigma}$',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax1.transAxes)

ax2.text(215, 3e-4, '*',  verticalalignment='top', horizontalalignment='right',
         multialignment="left")
ax3.text(190, 8e-4, '*',  verticalalignment='top', horizontalalignment='right',
     multialignment="left")
"""
Create subfigure labels
"""
for n, ax in enumerate((ax1,ax2,ax3, ax4)):   
    ax.text(-0.35, 1, r'\textbf{'+ string.ascii_lowercase[n]+'}', transform=ax.transAxes, 
            size=8, weight='bold')
"""
Kurtosis plot style
"""
ax4.yaxis.set_ticks_position('both')
ax4.xaxis.set_ticks_position('both')
ax4.tick_params(which='both', axis="both", direction="in")
ax4.set_xscale('log')
ax4.set_ylim(0.5,2)
ax4.set_xlim(0.1,5)
ax4.set(xlabel=r'$\Updelta t$ (s) ',
        ylabel=r'$\kappa$ ')
ax4.legend(loc ='upper right', frameon = False,  handletextpad=0.1)
plt.tight_layout()
fig1.savefig(directory3 + '/AllGauss.pdf')