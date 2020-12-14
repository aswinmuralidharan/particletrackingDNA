#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:08:46 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import kde
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


def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff[diff == 0] = 'nan'
        yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

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

def powerfit(msd,a,b,fps=10):
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

def imsd(filename, mpp, fps, video, directory,min_time, max_time, plotindividual=False,  max_lagtime=100 ):
    statistic = 'msd'
    traj = pd.read_csv(filename)
    pos_columns = ['x', 'y']
    ids = []
    trajec = []
    msds = []
    alpha = []
    msdsuper = []
    msdsub = []
    alphasuper = []
    alphasub = []
    index=0
    for pid, ptraj in traj.groupby('particle'):
        pos = ptraj.set_index('frame')[pos_columns] * mpp
        pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1])).to_numpy()
        msdtemp = msd(pos, max_lagtime).replace(0, 'nan')
        msdtemp = msdtemp.dropna(axis = 'columns', how = 'all')
        if len(msdtemp.columns) == 1:
            r_squaredtemp = 0
        else:
            alphatemp, r_squaredtemp = powerfit(msdtemp, min_time, max_time)
        if r_squaredtemp > 0:
            if alphatemp[0] > 0:
                alpha.append(alphatemp)
                msds.append(msdtemp)
                trajec.append(trajectory(pos, pid, max_lagtime,video, pos_columns))
                ids.append(index)
                index += 1
                if alphatemp[0] > 1:
                    msdsuper.append(msdtemp)
                    alphasuper.append(alphatemp)
                elif alphatemp[0] < 1:
                    msdsub.append(msdtemp)
                    alphasub.append(alphatemp)
                if plotindividual==True:
                    y = msdtemp['msd']
                    x = msdtemp.index.values.astype('float64') / float(fps)
                    fig,ax = plt.subplots()
                    ax.loglog(x,y,'ro')
                    ax.loglog(x[20:50], power_law(x[20:50], *alphatemp),'b-')
                    ax.set_ylim((1e-3, 1))
                    ax.set(ylabel=r'$\langle \Updelta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
                            xlabel=r'$\mathrm{\tau}$ (s)')
                    ax.set_ylim((1e-4, 10))
                    ax.yaxis.set_ticks_position('both')
                    ax.xaxis.set_ticks_position('both')
                    ax.tick_params(which="both", axis="both", direction="in")
                    ax.set_aspect(0.25, adjustable='box')
                    ax.text(0.05, 0.95,  r'$\alpha$ = ' + str(alphatemp), verticalalignment='top', horizontalalignment='left',
                             transform=ax.transAxes)
                    plt.savefig(directory + '/'+str(video)+str(index)+'.pdf')
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
    if msdsuper:
        resultssuper = pd.concat(msdsuper, keys = ids)
        resultssuper = resultssuper.swaplevel(0, 1)[statistic].unstack()
        lagt = resultssuper.index.values.astype('float64') / float(fps)
        resultssuper.set_index(lagt, inplace=True)
        resultssuper.index.name = 'lag time [s]'
        alphasuper = pd.DataFrame(alphasuper)
    else:
        resultssuper = []
    if msdsub:
        resultssub = pd.concat(msdsub, keys = ids)
        resultssub = resultssub.swaplevel(0, 1)[statistic].unstack()
        lagt = resultssub.index.values.astype('float64') / float(fps)
        resultssub.set_index(lagt, inplace=True)
        resultssub.index.name = 'lag time [s]'
        alphasub = pd.DataFrame(alphasub)
    else:
        resultssub = []   
    return resultsmsd, resultstraj, alpha,resultssuper, resultssub, alphasuper, alphasub


"""
Main code
--------------------------------------------------------------------------------
"""
mpp = 0.16
fps = 10
max_lagtime = 100
bpc = ['100bp' , '250bp', '500bp']
bpcs = ['100 bp' , '250 bp', '500 bp']
fig1, ax = plt.subplots(2,3, figsize=(3.375*2,3.375*4/3))
fig2, ax2 = plt.subplots(2,1, figsize=(3.375,3.375*2))
i=0
for bp in bpc:
    Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tf'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures' 
    currentlength = 0
    video = 0
    Trajcollected = []
    alphacollected = []
    msdcolumnscollected = []
    bps=bpcs[i]
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            meansquaredis, traj, alpha, resultssuper, resultssub, alphasupert,alphasubt = imsd(os.path.join(directory, filename), mpp, fps, video, directory2, 0, 8,False, max_lagtime)
            MSDcolumns = [i + currentlength for i in range(1, len(meansquaredis.columns) + 1)]
            if currentlength == 0:
                MSDcollected_df = pd.DataFrame(meansquaredis, columns= np.arange(0,len(meansquaredis.columns) + 1))
            else:
                MSDcollected_df[MSDcolumns] = meansquaredis
            Trajcollected.append(traj)
            alphacollected.append(alpha)
            video += 1
            currentlength = len(MSDcollected_df.columns)
            msdcolumnscollected.append(MSDcolumns)
    MSDcollected_df = MSDcollected_df.dropna(axis = 'columns', how = 'all')
    Trajcollected_df = pd.concat(Trajcollected)
    Trajcollected_df = Trajcollected_df.dropna(axis = 'columns', how = 'all')
    Alphacollected_df = pd.concat(alphacollected)
    totaltracks = len(MSDcollected_df.columns)
    alpha_his250 = Alphacollected_df[1].to_numpy()
    alpha_his = Alphacollected_df[0].to_numpy()
    k = kde.gaussian_kde([np.log10(alpha_his250),alpha_his])
    xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    his1 = ax[0,i].contourf(xi,yi,zi.reshape(xi.shape), 200,cmap = 'Spectral_r', alpha = 1)
    ax[0,i].scatter(np.log10(alpha_his250), alpha_his, marker = '.', color = 'k', s= 1 )
    ax[0,i].set(xlabel=r'$\log (D_{\mathrm{app}})$ ',
               ylabel=r'$\alpha_1$')
    ax[0,i].set_yticks(np.arange(0, 2.01, step=0.5))
    ax[0,i].set_xticks(np.arange(-4, 0.01, step=1))
    ax[0,i].set_xlim(-4,0)
    ax[0,i].set_ylim(0,2)
    ax[0,i].yaxis.set_ticks_position('both')
    ax[0,i].xaxis.set_ticks_position('both')
    ax[0,i].tick_params(which='both', axis="both", direction="in")
    ax[0,i].text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
             multialignment="left",
             transform=ax[0,i].transAxes, color = 'white')
    i+=1


i=0
for bp in bpc:
    Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
    for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
    directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
    directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures' 
    currentlength = 0
    video = 0
    Trajcollected = []
    alphacollected = []
    msdcolumnscollected = []
    bps = bpcs[i]
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            meansquaredis, traj, alpha, resultssuper, resultssub, alphasupert,alphasubt = imsd(os.path.join(directory, filename), mpp, fps, video, directory2, 20, 50,False, max_lagtime)
            MSDcolumns = [i + currentlength for i in range(1, len(meansquaredis.columns) + 1)]
            if currentlength == 0:
                MSDcollected_df = pd.DataFrame(meansquaredis, columns= np.arange(0,len(meansquaredis.columns) + 1))
            else:
                MSDcollected_df[MSDcolumns] = meansquaredis
            Trajcollected.append(traj)
            alphacollected.append(alpha)
            video += 1
            currentlength = len(MSDcollected_df.columns)
            msdcolumnscollected.append(MSDcolumns)
    MSDcollected_df = MSDcollected_df.dropna(axis = 'columns', how = 'all')
    Trajcollected_df = pd.concat(Trajcollected)
    Trajcollected_df = Trajcollected_df.dropna(axis = 'columns', how = 'all')
    Alphacollected_df = pd.concat(alphacollected)
    totaltracks = len(MSDcollected_df.columns)
    alpha_his250 = Alphacollected_df[1].to_numpy()
    alpha_his = Alphacollected_df[0].to_numpy()
    k = kde.gaussian_kde([np.log10(alpha_his250),alpha_his])
    xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    his1 = ax[1,i].contourf(xi,yi,zi.reshape(xi.shape), 200,cmap = 'Spectral_r', alpha = 1)
    ax[1,i].scatter(np.log10(alpha_his250), alpha_his, marker = '.', color = 'k', s= 1 )
    ax[1,i].set(xlabel=r'$\log (D_{\mathrm{app}})$ ',
               ylabel=r'$\alpha_2$')
    ax[1,i].set_yticks(np.arange(0, 2.01, step=0.5))
    ax[1,i].set_xticks(np.arange(-4, 0.01, step=1))
    ax[1,i].set_xlim(-4,0)
    ax[1,i].set_ylim(0,2)
    ax[1,i].yaxis.set_ticks_position('both')
    ax[1,i].xaxis.set_ticks_position('both')
    ax[1,i].tick_params(which='both', axis="both", direction="in")
    ax[1,i].text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
             multialignment="left",
             transform=ax[1,i].transAxes, color = 'white')
    i+=1

axlab =0
for n, row in enumerate(ax): 
    for j, ax in enumerate(row):
        ax.text(-0.2, 1, r'\textbf{'+ string.ascii_lowercase[axlab]+'}', transform=ax.transAxes, 
            size=8, weight='bold')
        axlab+=1
plt.tight_layout()
fig1.savefig(directory3 + '/contour2.pdf')
