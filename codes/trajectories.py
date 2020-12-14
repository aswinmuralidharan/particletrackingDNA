#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:30:17 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.stats import kde


plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')


def _msd_iter(pos, lagtimes):
    for lt in lagtimes:
        diff = pos[lt:] - pos[:-lt]
        diff[diff == 0] = 'nan'
        yield np.concatenate((np.nanmean(diff, axis=0), np.nanmean(diff ** 2, axis=0)))

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    return lc

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

def powerfit(msd,fps=10):
    y = msd['msd']
    x = msd.index.values.astype('float64') / float(fps)
    pars, cov = curve_fit(f = power_law, xdata = x[20:50], ydata = y[20:50], p0=[0, 0], bounds=(-np.inf, np.inf))
    result = pars
    residuals = y[20:50] - power_law(x[20:50], *pars)
    ss_res = np.sum(residuals**2)
    #total sum of squares SStot
    ss_tot = np.sum((y[20:50]-np.mean(y[20:50]))**2)
    #R-squared 
    r_squared = 1.0 - (ss_res / ss_tot)
    return result, r_squared

def imsd(filename, mpp, fps, video, directory, plotindividual=False, max_lagtime=100):
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
            alphatemp, r_squaredtemp = powerfit(msdtemp)
        if r_squaredtemp > 0.9:
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
                    ax.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
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

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = '500bp'  # Base pair to process
bps = '500 bp'
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
currentlengthsuper = 0
currentlengthsub = 0
video = 0
Trajcollected = []
alphacollected = []
alphasuper = []
alphasub = []
msdcolumnscollected = []

"""
Obtain the mean square displacement and trajectory translated to zero as two 
dataframes
--------------------------------------------------------------------------------
"""

for filename in os.listdir(directory):
    if not filename.startswith('.'):
        meansquaredis, traj, alpha, resultssuper, resultssub, alphasupert,alphasubt = imsd(os.path.join(directory, filename), mpp, fps, video, directory2,False, max_lagtime)
        MSDcolumns = [i + currentlength for i in range(1, len(meansquaredis.columns) + 1)]
        if currentlength == 0:
            MSDcollected_df = pd.DataFrame(meansquaredis, columns= np.arange(0,len(meansquaredis.columns) + 1))
        else:
            MSDcollected_df[MSDcolumns] = meansquaredis
        if type(resultssuper) == pd.core.frame.DataFrame:
            MSDsupercolumns = [i + currentlengthsuper for i in range(1, len(resultssuper.columns) + 1)]
            alphasuper.append(alphasupert)
            if currentlengthsuper == 0:
                MSDsuper_df = pd.DataFrame(resultssuper, columns= np.arange(0,len(resultssuper.columns) + 1))
            else:
                MSDsuper_df[MSDsupercolumns] = resultssuper
            currentlengthsuper = len(MSDsuper_df.columns)
        if type(resultssub) == pd.core.frame.DataFrame:
            MSDsubcolumns = [i + currentlengthsub for i in range(1, len(resultssub.columns) + 1)]
            alphasub.append(alphasubt)
            if currentlengthsub == 0:
                MSDsub_df = pd.DataFrame(resultssub, columns= np.arange(0,len(resultssub.columns) + 1))
            else:
                MSDsub_df[MSDsubcolumns] = resultssub
            currentlengthsub = len(MSDsub_df.columns)
        Trajcollected.append(traj)
        alphacollected.append(alpha)
        video += 1
        currentlength = len(MSDcollected_df.columns)
        msdcolumnscollected.append(MSDcolumns)
MSDcollected_df = MSDcollected_df.dropna(axis = 'columns', how = 'all')
Trajcollected_df = pd.concat(Trajcollected)
Trajcollected_df = Trajcollected_df.dropna(axis = 'columns', how = 'all')
Alphacollected_df = pd.concat(alphacollected)
Alphacollected_super_df = pd.concat(alphasuper)
Alphacollected_sub_df = pd.concat(alphasub)
MSDsuper_df = MSDsuper_df.dropna(axis = 'columns', how = 'all')
MSDsub_df = MSDsub_df.dropna(axis = 'columns', how = 'all')
totaltracks = len(MSDcollected_df.columns)
totalsub = len(MSDsub_df.columns)
totalsup = len(MSDsuper_df.columns)

"""
Plotting the mean square displacement as ensemble average and individual
--------------------------------------------------------------------------------
"""

fig1, ax1 = plt.subplots()
ys = np.transpose(MSDcollected_df.to_numpy())
xs = [np.transpose(MSDcollected_df.index.to_numpy())]*ys.shape[1]
norm1 = mcolors.Normalize(0, 2)
lc1 = multiline(xs,ys, Alphacollected_df[0].to_numpy(), cmap='viridis',norm=norm1, ax=ax1)
ax1cb = fig1.colorbar(lc1)
ax1cb.ax.tick_params(axis='y', direction='in')
ax1cb.ax.set_ylim(0,2)
ax1cb.set_clim(0,2)
ax1cb.set_ticks(np.arange(0, 2.2, step=0.4))
ax1cb.set_label(r'$\alpha$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax1.set_ylim((1e-4, 10))
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which="both", axis="both", direction="in")
ax1.set_aspect(0.5, adjustable='box')
ax1.text(0.05, 0.95, bps + '\n$n$ = ' + str(totaltracks), verticalalignment='top', horizontalalignment='left',
         multialignment="left",
         transform=ax1.transAxes)
#fig1.savefig(directory3 + '/MSD.eps')

"""
Plotting subdiffusive 
"""

fig2, ax2 = plt.subplots()
ysub = np.transpose(MSDsub_df.to_numpy())
xsub = [np.transpose(MSDsub_df.index.to_numpy())]*ys.shape[1]
norm1 = mcolors.Normalize(0, 2)
lc2 = multiline(xsub,ysub, Alphacollected_sub_df[0].to_numpy(), cmap='viridis',norm=norm1, ax=ax2)
ax2cb = fig2.colorbar(lc2)
ax2cb.ax.tick_params(axis='y', direction='in')
ax2cb.ax.set_ylim(0,2)
ax2cb.set_clim(0,2)
ax2cb.set_ticks(np.arange(0, 2.2, step=0.4))
ax2cb.set_label(r'$\alpha$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax2.set_ylim((1e-4, 10))
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.tick_params(which="both", axis="both", direction="in")
ax2.set_aspect(0.5, adjustable='box')
ax2.text(0.05, 0.95, bps + '\nSubdiffusive \n$n$ = ' + str(totalsub), verticalalignment='top', horizontalalignment='left',
         multialignment="left",
         transform=ax2.transAxes)
#fig2.savefig(directory3 + '/MSDsub.eps')

"""
Plotting superdiffusive 
"""

fig3, ax3 = plt.subplots()
ysup = np.transpose(MSDsuper_df.to_numpy())
xsup = [np.transpose(MSDsuper_df.index.to_numpy())]*ys.shape[1]
norm1 = mcolors.Normalize(0, 2)
lc3 = multiline(xsup,ysup, Alphacollected_super_df[0].to_numpy(), cmap='viridis',norm=norm1, ax=ax3)
ax3cb = fig3.colorbar(lc2)
ax3cb.ax.tick_params(axis='y', direction='in')
ax3cb.ax.set_ylim(0,2)
ax3cb.set_clim(0,2)
ax3cb.set_ticks(np.arange(0, 2.2, step=0.4))
ax3cb.set_label(r'$\alpha$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax3.set_ylim((1e-4, 10))
ax3.yaxis.set_ticks_position('both')
ax3.xaxis.set_ticks_position('both')
ax3.tick_params(which="both", axis="both", direction="in")
ax3.set_aspect(0.5, adjustable='box')
ax3.text(0.05, 0.95, bps + '\nSuperdiffusive \n$n$ = ' + str(totalsup), verticalalignment='top', horizontalalignment='left',
         multialignment="left",
         transform=ax3.transAxes)
#fig3.savefig(directory3 + '/MSDsup.eps')

"""
Plotting the trajectories colored with time
--------------------------------------------------------------------------------
"""

fig4, ax4 = plt.subplots()
ax4.axhline(y=0, color='k', linestyle='dashdot')
ax4.axvline(x=0, color='k', linestyle='dashdot')
t = np.arange(0, 10.1, 0.1)
for id, traj in Trajcollected_df.groupby(['particle', 'video']):
    x = traj['x'].head(100)
    y = traj['y'].head(100)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2)
    line = ax4.add_collection(lc)
    del x, y
fig4.colorbar(line, ax=ax4, label = '$t$ (s)').ax.tick_params(axis='y', direction='in')
bounds = 2.5
ax4.set_xlim(-bounds, bounds)
ax4.set_ylim(-bounds, bounds)
ax4.set(ylabel=r'y ($\mathrm{\mu}$m)',
        xlabel=r'x ($\mathrm{\mu}$m)')
ax4.set_aspect('equal', adjustable='box')
plt.minorticks_on()
ax4.yaxis.set_ticks_position('both')
ax4.xaxis.set_ticks_position('both')
ax4.tick_params(which='both', axis="both", direction="in")
ax4.text(0.05, 0.95, bps + '\n$n$ = ' + str(totaltracks), verticalalignment='top', horizontalalignment='left',
         multialignment="left",
         transform=ax4.transAxes)
#fig4.savefig(directory3 + '/Traj.eps')

"""
Histogram diffusion coefficient
--------------------------------------------------------------------------------
"""

fig5,ax5 = plt.subplots(figsize=(4,4))
alpha_his250 = Alphacollected_df[1].to_numpy()
weights = np.ones_like(alpha_his250)/len(alpha_his250)
ax5.hist(np.log10(Alphacollected_df[1].to_numpy()),bins = np.arange(-6,0,0.25), edgecolor='black',color = 'white', weights = weights, rwidth=1)
ax5.set_xlim(-4,0)
ax5.set_ylim(0,0.3)
#ax5.set_xscale('log')
ax5.yaxis.set_ticks_position('both')
ax5.xaxis.set_ticks_position('both')
ax5.tick_params(which='both', axis="both", direction="in")
#ax5.set_aspect(15, adjustable='box')
ax5.text(0.95, 0.95, bps , verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax5.transAxes)
ax5.set(xlabel=r'$\log (D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$)) ',
        ylabel=r'Probability')
ax5.set_yticks(np.arange(0, 0.35, step=0.05))
plt.tight_layout()
fig5.savefig(directory3 + '/HistoD.eps')

"""
Histogram alpha 
--------------------------------------------------------------------------------
"""

fig6,ax6 = plt.subplots(figsize=(4,4))
alpha_his = Alphacollected_df[0].to_numpy()
weights = np.ones_like(alpha_his)/len(alpha_his)
ax6.hist(Alphacollected_df[0],bins = np.arange(0,2,0.1), edgecolor='black',color = 'white', weights = weights, rwidth=1)
ax6.set_xlim(0,2)
ax6.set_ylim(0,0.25)
ax6.yaxis.set_ticks_position('both')
ax6.xaxis.set_ticks_position('both')
ax6.tick_params(which='both', axis="both", direction="in")
ax6.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax6.transAxes)
ax6.set(xlabel=r'$\alpha_2$ ',
        ylabel=r'$P(\alpha_2)$')
ax6.set_yticks(np.arange(0, 0.3, step=0.05))
plt.tight_layout()
#fig5.savefig(directory3 + '/Histoalpha.eps')

fig7,ax7 = plt.subplots(figsize=(4,3.5))

#his1 = ax7.hist2d(np.log10(alpha_his250), alpha_his, [np.arange(-4,0.1,0.4), np.arange(0,2.1,0.2)], cmap = 'Spectral_r', density = True, alpha = 0.5, norm = LogNorm(clip = False))


k = kde.gaussian_kde([np.log10(alpha_his250),alpha_his])
xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
his1 = ax7.contourf(xi,yi,zi.reshape(xi.shape), 200,cmap = 'RdGy_r', alpha = 1)
ax4cb = fig7.colorbar(his1)
contours = ax7.contour(xi, yi, zi.reshape(xi.shape), 6, colors='black')
#plt.clabel(contours, inline=True, fontsize=8)
ax7.scatter(np.log10(alpha_his250), alpha_his, marker = '.', color = 'k', s= 1 )
ax7.set(xlabel=r'$\log (D_{\mathrm{app}})$ ',
        ylabel=r'$\alpha_2$')
ax7.set_yticks(np.arange(0, 2.01, step=0.5))
ax7.set_xticks(np.arange(-4, 0.01, step=1))
ax7.set_xlim(-4,0)
ax7.set_ylim(0,2)
ax7.yaxis.set_ticks_position('both')
ax7.xaxis.set_ticks_position('both')
ax7.tick_params(which='both', axis="both", direction="in")
ax7.text(0.95, 0.95, bps, verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=ax7.transAxes, color = 'white')
ax4cb.ax.tick_params(axis='y', direction='in')
ax4cb.ax.set_ylim(0,1)
ax4cb.set_clim(0,1)
ax4cb.set_ticks(np.arange(0, 1.1, step=0.2))
plt.tight_layout()


"""
Save the files
--------------------------------------------------------------------------------
"""
"""
Total width of the latex image is 7.0586 inches
"""
#MSDcollected_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/' +  'MSDcollected.csv')
#MSDsub_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/' +  'MSDsub.csv')
#MSDsuper_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/MSDcollected/' +  'MSDsuper.csv')
Trajcollected_df.to_csv(Filepath + '/E_output_data/' + str(bp) + '/Trajcollected/' + 'Trajcollected.csv')
del directory, directory2, dirs, filename, fps, id, max_lagtime,  mpp, points, segments,t, totaltracks, traj,video, bp, bps,bounds,alpha, MSDcolumns, Trajcollected, Filepath,alphacollected,currentlength,meansquaredis
del MSDsubcolumns, MSDsupercolumns, currentlengthsub, currentlengthsuper,weights, xs, ys, alphasubt, alphasupert, totalsub, totalsup, xsub, xsup, ysub, ysup, alphasub, alphasuper, resultssub, resultssuper