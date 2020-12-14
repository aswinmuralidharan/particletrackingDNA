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
import string
from matplotlib.ticker import MultipleLocator

plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=10)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')

def fitGaussian(x,a,sigma):
    return (a*np.exp(-((x)**2/(2*sigma))))

def fitexp(x,a,sigma):
    return (a*np.exp(-((x)/(2*sigma))))

def vanhove(filename, lagtime = 1, maxtime = 100, mpp=0.16):
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
    x_4 = np.power(data,4)
    x_2 = np.power(data,2)
    kurtosis = np.mean(x_4)/(np.mean(x_2)**2)-3
    return kurtosis

fig1,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(3.375*2,3.375*1.618))


       
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
        vanho = vanhove(os.path.join(directory, filename)) 
        
x =  vanho['x'].append(vanho['y']).reset_index()       
x_his = np.abs(x.to_numpy()/0.001)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,300,20), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
bin_centres2 = bin_centres[np.where(bin_centres<=100)]
counts2 = counts[np.where(bin_centres<=100)]
bin_centres3 = np.arange(0,200)
popt, pcov = curve_fit(fitGaussian, bin_centres2, counts2, [0.1,30])

bin_centres4 = bin_centres[np.where(bin_centres >=100)]
counts4 = counts[np.where((bin_centres >=100))]
bin_centres5 = np.arange(0,250)
popt2, pcov2 = curve_fit(fitexp, bin_centres4, counts4, [0.1,30])

ax1.plot(bin_centres3, fitGaussian(bin_centres3,*popt), 'k', linewidth=1, label = 'Gaussian fit')
ax1.plot(bin_centres5, fitexp(bin_centres5,*popt2), 'k-.', linewidth=1, label = 'Gaussian fit')

ax1.plot(bin_centres,counts, marker = 'o', markerfacecolor = 'lightgray', markeredgecolor = 'k', linestyle = 'None', label = 'Experiment' )
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which='both', axis="both", direction="in")
ax1.set_yscale('log')
ax1.set_ylim(1e-5,0.1)
ax1.set_xlim(0,250)
ax1.set(xlabel=r'$\mid\Delta x\mid$ (nm) ',
        ylabel=r'G$(\mid\Delta x\mid, \Delta t)$ ')

#ax1.legend(frameon = False, loc = 'upper right')
ax1.text(0.95, 0.95, bps + '\n'+ r'$\Delta t = 0.1$ s',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax1.transAxes)


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
        vanho = vanhove(os.path.join(directory, filename)) 
        
x =  vanho['x'].append(vanho['y']).reset_index()       
x_his = np.abs(x.to_numpy()/0.001)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,300,20), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
bin_centres2 = bin_centres[np.where(bin_centres<=100)]
counts2 = counts[np.where(bin_centres<=100)]
bin_centres3 = np.arange(0,200)
popt, pcov = curve_fit(fitGaussian, bin_centres2, counts2, [0.1,30])

bin_centres4 = bin_centres[np.where((bin_centres >=100))]
counts4 = counts[np.where((bin_centres >=100))]
bin_centres5 = np.arange(0,250)
popt2, pcov2 = curve_fit(fitexp, bin_centres4, counts4, [0.1,30])

ax2.plot(bin_centres3, fitGaussian(bin_centres3,*popt), 'k', linewidth=1, label = 'Gaussian fit')
ax2.plot(bin_centres5, fitexp(bin_centres5,*popt2), 'k-.', linewidth=1, label = 'Gaussian fit')

ax2.plot(bin_centres,counts, marker = 'o', markerfacecolor = 'lightgray', markeredgecolor = 'k', linestyle = 'None', label = 'Experiment' )
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.tick_params(which='both', axis="both", direction="in")
ax2.set_yscale('log')
ax2.set_ylim(1e-5,0.1)
ax2.set_xlim(0,250)
ax2.set(xlabel=r'$\mid\Delta x\mid$ (nm) ',
        ylabel=r'G$(\mid\Delta x\mid, \Delta t)$ ')

#ax1.legend(frameon = False, loc = 'upper right')
ax2.text(0.95, 0.95, bps + '\n'+ r'$\Delta t = 0.1$ s',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax2.transAxes)

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
        vanho = vanhove(os.path.join(directory, filename)) 
        
x =  vanho['x'].append(vanho['y']).reset_index()       
x_his = np.abs(x.to_numpy()/0.001)
counts,bin_edges = np.histogram(x_his,bins = np.arange(0,300,20), density = True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
bin_centres2 = bin_centres[np.where(bin_centres<=100)]
counts2 = counts[np.where(bin_centres<=100)]
bin_centres3 = np.arange(0,200)
popt, pcov = curve_fit(fitGaussian, bin_centres2, counts2, [0.1,30])

bin_centres4 = bin_centres[np.where((bin_centres >=100))]
counts4 = counts[np.where((bin_centres >=100))]
bin_centres5 = np.arange(0,250)
popt2, pcov2 = curve_fit(fitexp, bin_centres4, counts4, [0.1,30])

ax3.plot(bin_centres3, fitGaussian(bin_centres3,*popt), 'k', linewidth=1, label = 'Gaussian fit')
ax3.plot(bin_centres5, fitexp(bin_centres5,*popt2), 'k-.', linewidth=1, label = 'Gaussian fit')

ax3.plot(bin_centres,counts, marker = 'o', markerfacecolor = 'lightgray', markeredgecolor = 'k', linestyle = 'None', label = 'Experiment' )
ax3.yaxis.set_ticks_position('both')
ax3.xaxis.set_ticks_position('both')
ax3.tick_params(which='both', axis="both", direction="in")
ax3.set_yscale('log')
ax3.set_ylim(1e-5,0.1)
ax3.set_xlim(0,250)
ax3.set(xlabel=r'$\mid\Delta x\mid$ (nm) ',
        ylabel=r'G$(\mid\Delta x\mid, \Delta t)$ ')

#ax1.legend(frameon = False, loc = 'upper right')
ax3.text(0.95, 0.95, bps + '\n' +r'$\Delta t = 0.1$ s',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax3.transAxes)

ax3.text(0.38, 0.25, r'G$\sim e^{-\mid \Delta x \mid ^2}$',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax3.transAxes)
ax3.text(0.9, 0.55, r'G$\sim e^{-\mid \Delta x \mid}$',  verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax3.transAxes)

for n, ax in enumerate((ax1,ax2,ax3)):   
    ax.text(-0.3, 1, string.ascii_lowercase[n], transform=ax.transAxes, 
            size=12, weight='bold')
    ax.xaxis.set_minor_locator(MultipleLocator(10))

bp = '100bp'  # Base pair to process
bps = '100 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
kurt = []
for tau in np.arange(1,51):
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            vanho = vanhove(os.path.join(directory, filename),tau)         
    x =  vanho['x'].append(vanho['y']).reset_index()       
    x_his = np.abs(x.to_numpy()/0.001)
    kurt.append(kurtosis(x_his))

#fig2, ax4 = plt.subplots(1,1, figsize=(10/3,10/3))
ax4.plot(np.arange(1,51)/10, kurt, marker = 'o', markerfacecolor = '#fc8d59', markeredgecolor = 'k', linestyle = 'None', label = '100 bp')
del kurt

bp = '250bp'  # Base pair to process
bps = '250 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
kurt = []
for tau in np.arange(1,51):
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            vanho = vanhove(os.path.join(directory, filename),tau)         
    x =  vanho['x'].append(vanho['y']).reset_index()       
    x_his = np.abs(x.to_numpy()/0.001)
    kurt.append(kurtosis(x_his))
ax4.plot(np.arange(1,51)/10, kurt, marker = 'o', markerfacecolor = '#ffffbf', markeredgecolor = 'k', linestyle = 'None', label = '250 bp')
del kurt

bp = '500bp'  # Base pair to process
bps = '500 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
kurt = []
for tau in np.arange(1,51):
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            vanho = vanhove(os.path.join(directory, filename),tau)         
    x =  vanho['x'].append(vanho['y']).reset_index()       
    x_his = np.abs(x.to_numpy()/0.001)
    kurt.append(kurtosis(x_his))
ax4.plot(np.arange(1,51)/10, kurt, marker = 'o', markerfacecolor = '#91bfdb', markeredgecolor = 'k', linestyle = 'None', label = '500 bp')
ax4.yaxis.set_ticks_position('both')
ax4.xaxis.set_ticks_position('both')
ax4.tick_params(which='both', axis="both", direction="in")
ax4.set_xscale('log')
ax4.set_ylim(0.5,2)
ax4.set_xlim(0.1,5)
ax4.set(xlabel=r'$\Delta t$ (s) ',
        ylabel=r'$\kappa$ ')
ax4.legend(loc ='upper right', frameon = False,  handletextpad=0.1)
ax4.text(-0.3, 1, string.ascii_lowercase[3], transform=ax4.transAxes, 
            size=12, weight='bold')
ax4.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.tight_layout()
fig1.savefig(directory3 + '/AllGauss.pdf')