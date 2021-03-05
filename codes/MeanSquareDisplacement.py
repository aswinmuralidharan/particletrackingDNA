#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:51:55 2021

@author: aswinmuralidharan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import string
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib as mpl

plt.style.use('aswinplotstyle') # Custom plot style file. Comment out/use your own style file. 
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # LaTeX file path. Replace with your own pathfile.

""" Style parameters for the plots. """
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
       r'\usepackage{sansmath}',  
       r'\sansmath'              
       r"\usepackage{amsmath}"
       r"\usepackage{textgreek}"
       r"\usepackage{upgreek}"
]  
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif'
plt.rcParams['axes.linewidth'] = 1
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

""" Mean square displacement code """
def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff[diff == 0] = 'nan'
        yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

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
        
def msd(pos, max_lagtime):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals.
    Parameters
    ----------
    pos : Positions x, and y for individual particles 
    max_lagtime : intervals of frames out to which MSD is computed
    Returns
    -------
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)
    Notes
    -----
    Input units are microns and frames. Output units are microns and seconds."""

    max_lagtime = min(max_lagtime, len(pos) - 1)
    lagtimes = np.arange(1, max_lagtime + 1)
    pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]
    resultmsd = _msd_iter(pos, lagtimes)
    result = pd.DataFrame(resultmsd, columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    return result

""" General function for power law """
def power_law(x,alpha, A):
    return A*np.power(x, alpha)

""" 
Function for fitting power law. Input parameters are the mean square displacement dataframe and the frame rate of the camera. 
The two other input parameters are the first and last index of the data to fit. 
Choose the range of the range of the data to fit within the function at curve_fit. The R squared value of the fit is also computed 
"""
def powerfit(msd, a,b, fps=10):
    y = msd['msd']
    x = msd.index.values.astype('float64') / float(fps)
    pars, cov = curve_fit(f = power_law, xdata = x[a:b], ydata = y[a:b], p0=[0, 0], bounds=(-np.inf, np.inf))
    result = pars
    residuals = y[a:b] - power_law(x[a:b], *pars)
    ss_res = np.sum(residuals**2)
    #total sum of squares SStot
    ss_tot = np.sum((y[a:b]-np.mean(y[a:b]))**2)
    #R-squared 
    r_squared = 1.0 - (ss_res / ss_tot)
    return result, r_squared


"""
Function for computing the time averaged mean square displacement.
DataFrame of the mean square displacement, shifted trajectories and the anomalous exponent is returned.
"""
def imsd(filename, mpp, fps, video, directory, max_lagtime=100):
    statistic = 'msd'
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']
    ids = []
    trajec = []
    msds = []
    alpha = []
    alpha2 = []
    index=0
    """Looping through the dataframe with the position data"""
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        msdtemp = msd(pos, max_lagtime).replace(0, 'nan')
        msdtemp = msdtemp.dropna(axis = 'columns', how = 'all')
        """ Drop empty files to avoid errors """
        if len(msdtemp.columns) == 1:
            r_squaredtemp = 0
        else:
            """
            Make sure that the power law exponent is positive (at the start). Set a threshold for r squared 
            values to reject trajectories if needed
            """
            alphatemp, r_squaredtemp = powerfit(msdtemp, 0, 8)
            alphatemp2, r_squaredtemp2 = powerfit(msdtemp, 20,50)
        if r_squaredtemp > 0:
            if alphatemp[0] > 0:
                # Append results to list
                alpha.append(alphatemp)
                alpha2.append(alphatemp2)
                msds.append(msdtemp)
                trajec.append(trajectory(pos, pid, max_lagtime,video, pos_columns))
                ids.append(index)
                index += 1
                del msdtemp        
    resultsmsd = pd.concat(msds, keys=ids)
    resultsmsd = resultsmsd.swaplevel(0, 1)[statistic].unstack()
    lagt = resultsmsd.index.values.astype('float64') / float(fps)
    resultsmsd.set_index(lagt, inplace=True)
    resultsmsd.index.name = 'lag time [s]'
    resultstraj = pd.concat(trajec)
    time = resultstraj.index.values.astype('float64') / float(fps)
    resultstraj.set_index(time, inplace=True)
    resultstraj.index.name = 'Time [s]'
    alpha = pd.DataFrame(alpha)
    alpha2 = pd.DataFrame(alpha2)
    return resultsmsd, resultstraj, alpha, alpha2

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
i = 0
bpc = ['100bp', '250bp', '500bp']
bpcs = ['100 bp', '250 bp', '500 bp']
#bpc = ['500bp','MCF7500bp', 'MCF10A500bp']
#bpcs = ['CHO 500 bp','MCF7 500 bp', 'MCF10A 500 bp']

"""
Initialize plots. Change as necessary

"""
fig1, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(3.375*2,3.375*2.2/3))
ax= (ax1,ax2,ax3)
fig2, (ax4) = plt.subplots (1,1, figsize=(3.375*2/3,3.375*2.2/3)) 
colors = ['#fc8d59','#ffffbf', '#91bfdb']
mkr = ['^' , 'o', 's']
"""
Looping through the foldernames given in the list bpc
"""
for bp in bpc:
    """
    Create folders which arent there
    """
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tf'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures' 
    mpp = 0.16
    fps = 10
    max_lagtime = 100
    currentlength = 0
    video = 0
    Trajcollected = []
    alphacollected = []
    alphacollected2 = []
    msdcolumnscollected = []
    bps = bpcs[i]
    """
    Obtain the mean square displacement and trajectory translated to zero as two 
    dataframes
    """
    
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            meansquaredis, traj, alpha, alpha2 = imsd(os.path.join(directory, filename), mpp, fps, video, directory2, max_lagtime)
            MSDcolumns = [i + currentlength for i in range(1, len(meansquaredis.columns) + 1)]
            if currentlength == 0:
                MSDcollected_df = pd.DataFrame(meansquaredis, columns= np.arange(0,len(meansquaredis.columns) + 1))
            else:
                MSDcollected_df[MSDcolumns] = meansquaredis
            Trajcollected.append(traj)
            alphacollected.append(alpha)
            alphacollected2.append(alpha2)
            video += 1
            currentlength = len(MSDcollected_df.columns)
            msdcolumnscollected.append(MSDcolumns)
    MSDcollected_df = MSDcollected_df.dropna(axis = 'columns', how = 'all')
    Trajcollected_df = pd.concat(Trajcollected)
    Trajcollected_df = Trajcollected_df.dropna(axis = 'columns', how = 'all')
    Alphacollected_df = pd.concat(alphacollected)
    Alphacollected2_df = pd.concat(alphacollected2)
    totaltracks = len(MSDcollected_df.columns)
    """
    Plotting the mean square displacement as ensemble average and individual
    """
    
    ax[i].plot(MSDcollected_df, color = 'lightgray')
    ax[i].plot(MSDcollected_df.mean(axis = 1) , marker = 'o' , markeredgecolor = 'black' , markerfacecolor = '#228a8d', linestyle = 'None')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set(ylabel=r'$\langle \overline {\Updelta \mathbf{r}^2(\Updelta t)}\rangle$ ($\mathrm{\upmu}$m$^2$)',
            xlabel=r'$\Updelta t$ (s)')
    ax[i].set_ylim((1e-4, 10))
    ax[i].yaxis.set_ticks_position('both')
    ax[i].xaxis.set_ticks_position('both')
    ax[i].tick_params(which="both", axis="both", direction="in")
    ax[i].set_aspect(0.5, adjustable='box')
    ax[i].text(0.05, 0.95, bps + '\n$n$ = ' + str(totaltracks), verticalalignment='top', horizontalalignment='left',
             multialignment="left",
             transform=ax[i].transAxes)
    ax[i].text(-0.23, 1, r'\textbf{'+ string.ascii_uppercase[i]+'}', transform=ax[i].transAxes, 
            size=8, weight='bold')
    ax[i].xaxis.set_major_locator(LogLocator(base = 10, numticks =5))
    locmin = LogLocator(base=10.0,subs=tuple(np.arange(0.1, 1, 0.1)),numticks=5)
    ax[i].xaxis.set_minor_locator(locmin)
    ax[i].xaxis.set_minor_formatter(NullFormatter())
    
    ax4.plot(MSDcollected_df.mean(axis = 1) , marker = mkr[i] , markeredgecolor = 'black' , markerfacecolor = colors[i], linestyle = 'None',label = bpcs[i])
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set(ylabel=r'$\langle \overline {\Updelta \mathbf{r}^2(\Updelta t)}\rangle$ ($\mathrm{\upmu}$m$^2$)',
            xlabel=r'$\Updelta t$ (s)')
    ax4.set_ylim((1e-3, 0.1))
    ax4.yaxis.set_ticks_position('both')
    ax4.xaxis.set_ticks_position('both')
    ax4.tick_params(which="both", axis="both", direction="in")
    #ax4.set_aspect(0.5, adjustable='box')
    ax4.legend(loc ='upper left', frameon = False,  handletextpad=0.1)
             
    """ Print the early and late diffusion coefficient and exponent. """
    x = np.arange(0,100)/10
    y = np.array(MSDcollected_df.mean(axis=1))
    pars1, cov1 = curve_fit(f = power_law, xdata = x[20:50], ydata = y[20:50], p0=[0, 0], bounds=(-np.inf, np.inf))
    #print(pars1)
    pars2, cov2 = curve_fit(f = power_law, xdata = x[0:8], ydata = y[0:8], p0=[0, 0], bounds=(-np.inf, np.inf))
    #print(pars2)
    print(Alphacollected_df.mean())
    print(Alphacollected_df.sem())
    print(Alphacollected2_df.mean())
    print(Alphacollected2_df.sem())
    i+=1
fig1.tight_layout()
fig1.savefig(directory3 + '/MSD.png',dpi=300)
fig2.savefig(directory3 + '/MSDMean.png',dpi=300)