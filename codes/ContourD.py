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
from pylab import setp
from matplotlib.ticker import MaxNLocator
from scipy import stats
from operator import add

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

def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['medians'][1], color='red')
    

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
fig2, ax2 = plt.subplots(1,2, figsize=(3.375*2,3.375*0.6))
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
    D_his = Alphacollected_df[1].to_numpy()
    alpha_his = Alphacollected_df[0].to_numpy()
    k = kde.gaussian_kde([np.log10(D_his),alpha_his])
    xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    his1 = ax[0,i].contourf(xi,yi,zi.reshape(xi.shape), 200,cmap = 'Spectral_r', alpha = 1)
    ax[0,i].scatter(np.log10(D_his), alpha_his, marker = '.', color = 'k', s= 1 )
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
    if i == 0 :
        D100_1 = D_his
        A100_1 = alpha_his
    elif i == 1 :
        D250_1 = D_his
        A250_1 = alpha_his
    elif i == 2 :
        D500_1 = D_his
        A500_1 = alpha_his
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
    D_his = Alphacollected_df[1].to_numpy()
    alpha_his = Alphacollected_df[0].to_numpy()
    k = kde.gaussian_kde([np.log10(D_his),alpha_his])
    xi, yi = np.mgrid[-4:0.1:0.05, 0:2.1:0.025]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    his1 = ax[1,i].contourf(xi,yi,zi.reshape(xi.shape), 200,cmap = 'Spectral_r', alpha = 1)
    ax[1,i].scatter(np.log10(D_his), alpha_his, marker = '.', color = 'k', s= 1 )
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
    if i == 0 :
        D100_2 = D_his
        A100_2 = alpha_his
    elif i == 1 :
        D250_2 = D_his
        A250_2 = alpha_his
    elif i ==2 :
        D500_2 = D_his
        A500_2 = alpha_his
    i+=1

axlab =0
for n, row in enumerate(ax): 
    for j, ax in enumerate(row):
        ax.text(-0.2, 1, r'\textbf{'+ string.ascii_lowercase[axlab]+'}', transform=ax.transAxes, 
            size=8, weight='bold')
        axlab+=1
fig1.tight_layout()
fig1.savefig(directory3 + '/contour2.pdf')

bp1 = ax2[0].boxplot([D100_1, D100_2], positions = [1, 2], widths = 0.6, showfliers = False)
setBoxColors(bp1)

bp1 = ax2[0].boxplot([D250_1, D250_2], positions = [4, 5], widths = 0.6, showfliers = False)
setBoxColors(bp1)

bp1 = ax2[0].boxplot([D500_1, D500_2], positions = [7, 8], widths = 0.6, showfliers = False)
setBoxColors(bp1)

ax2[0].yaxis.set_ticks_position('both')
ax2[0].xaxis.set_ticks_position('both')
ax2[0].tick_params(which='both', axis="both", direction="in")
ax2[0].set_yscale('log')
ax2[0].set_ylim(1e-4,0.1)
ax2[0].set_xticklabels(['100 bp', '250 bp', '500 bp'])
ax2[0].set_xticks([1.5, 4.5, 7.5])
ax2[0].set(ylabel=r'$D_{\mathrm{app}}$ ($\mathrm{\upmu}$m$^2$s$^{-\alpha}$)')
ax2[0].set_xlim(0,9)
bp2 = ax2[1].boxplot([A100_1, A100_2], positions = [1, 2], widths = 0.6, showfliers = False)
setBoxColors(bp2)

bp2 = ax2[1].boxplot([A250_1, A250_2], positions = [4, 5], widths = 0.6, showfliers = False)
setBoxColors(bp2)

bp2 = ax2[1].boxplot([A500_1, A500_2], positions = [7, 8], widths = 0.6, showfliers = False)
setBoxColors(bp2)

ax2[1].yaxis.set_ticks_position('both')
ax2[1].xaxis.set_ticks_position('both')
ax2[1].tick_params(which='both', axis="both", direction="in")
ax2[1].set_yscale('linear')
ax2[1].set_ylim(0,2)
ax2[1].set_xticklabels(['100 bp', '250 bp', '500 bp'])
ax2[1].set_xticks([1.5, 4.5, 7.5])
ax2[1].set(ylabel=r'$\alpha$')
ax2[1].set_xlim(0,9)
hB, = ax2[1].plot([1,1],'b-')
hR, = ax2[1].plot([1,1],'r-')
ax2[1].legend((hB, hR),(r'$0<\Updelta t\mathrm{ \: (s) }<1$', r'$1<\Updelta t \mathrm{ \: (s) }<10$'),frameon = False,loc ='upper right')
hB.set_visible(False)
hR.set_visible(False)
hB, = ax2[0].plot([1,1],'b-')
hR, = ax2[0].plot([1,1],'r-')
ax2[0].legend((hB, hR),(r'$0<\Updelta t\mathrm{ \: (s) }<1$', r'$1<\Updelta t \mathrm{ \: (s) }<10$'),frameon = False,loc ='upper right')
hB.set_visible(False)
hR.set_visible(False)
ax2[1].yaxis.set_major_locator(MaxNLocator(nbins = 4))
for n,ax in enumerate(ax2): 
    ax.text(-0.13, 1, r'\textbf{'+ string.ascii_lowercase[n+3]+'}', transform=ax.transAxes, 
            size=8, weight='bold')
        
fig2.tight_layout()
fig2.savefig(directory3 + '/boxplot.pdf')
fig2.savefig(directory3 + '/boxplot.png',dpi=300)
X = np.array([1,3,5])

cagA100_1 =  np.size(A100_1[A100_1<0.4])/np.size(A100_1)*100
cagA250_1 =  np.size(A250_1[A250_1<0.4])/np.size(A250_1)*100
cagA500_1 =  np.size(A500_1[A500_1<0.4])/np.size(A500_1)*100

cagA100_2 =  np.size(A100_2[A100_2<0.4])/np.size(A100_2)*100
cagA250_2 =  np.size(A250_2[A250_2<0.4])/np.size(A250_2)*100
cagA500_2 =  np.size(A500_2[A500_2<0.4])/np.size(A500_2)*100

subA100_1 =  np.size(A100_1[A100_1<1])/np.size(A100_1)*100 - cagA100_1
subA250_1 =  np.size(A250_1[A250_1<1])/np.size(A250_1)*100 - cagA250_1
subA500_1 =  np.size(A500_1[A500_1<1])/np.size(A500_1)*100 - cagA500_1

subA100_2 =  np.size(A100_2[A100_2<1])/np.size(A100_2)*100 - cagA100_2
subA250_2 =  np.size(A250_2[A250_2<1])/np.size(A250_2)*100 - cagA250_2
subA500_2 =  np.size(A500_2[A500_2<1])/np.size(A500_2)*100 - cagA500_2

supA100_1 =  100 - cagA100_1 - subA100_1
supA250_1 =  100 - cagA250_1 - subA250_1
supA500_1 =  100 - cagA500_1 - subA500_1

supA100_2 =  100 - cagA100_2 - subA100_2
supA250_2 =  100 - cagA250_2 - subA250_2
supA500_2 =  100 - cagA500_2 - subA500_2

cag_1 = [cagA100_1, cagA250_1, cagA500_1]
cag_2 = [cagA100_2, cagA250_2, cagA500_2]

sub_1 = [subA100_1, subA250_1, subA500_1]
sub_2 = [subA100_2, subA250_2, subA500_2]

sup_1 = [supA100_1, supA250_1, supA500_1]
sup_2 = [supA100_2, supA250_2, supA500_2]

fig3, ax3 = plt.subplots(1,1, figsize=(3.375,3.375*0.6))
ind = np.arange(3)    # the x locations for the groups
ind2 = np.arange(0.25,3.25) 
width = 0.25  
p1 = plt.bar(ind, cag_1, width,fill=False, edgecolor='blue',hatch = '//', linewidth=1, label = 'Caged'+'\n'+r'$0<\alpha_1<0.4$', alpha=1)
p1 = plt.bar(ind2, cag_2, width,fill=False, edgecolor='red',hatch = '//', linewidth=1, label = 'Caged'+'\n'+r'$0<\alpha_2<0.4$', alpha=1)
p2 = plt.bar(ind, sub_1, width,fill=False, bottom = cag_1, edgecolor='blue',hatch = '..', linewidth=1, label = 'Subdiffusive'+'\n'+r'$0.4<\alpha_1<1$', alpha=1)
p2 = plt.bar(ind2, sub_2, width,fill=False, bottom = cag_2, edgecolor='red',hatch = '..', linewidth=1, label = 'Subdiffusive'+'\n'+r'$0.4<\alpha_2<1$', alpha=1)
p3 = plt.bar(ind, sup_1, width,fill=False, bottom = list(map(add, cag_1, sub_1)), edgecolor='blue', linewidth=1, label = 'Superdiffusive'+'\n'+r'$1<\alpha_1<2$', alpha=1)
p3 = plt.bar(ind2, sup_2, width,fill=False, bottom = list(map(add, cag_2, sub_2)), edgecolor='red', linewidth=1, label = 'Superdiffusive'+'\n'+r'$1<\alpha_2<2$', alpha=1)
ax3.set_ylim([0,110])
ax3.tick_params(which="both", axis="both", direction="in")
ax3.set_ylabel('Percentage')
plt.xticks([0.125, 1.125, 2.125], ('100 bp', '250 bp', '500 bp'))
ax3.legend(loc='upper center',frameon= False, bbox_to_anchor=(0.5, 1.4), ncol = 3)
fig3.savefig(directory3 + '/barchart.png', bbox_inches = "tight",dpi=300)
print(stats.ttest_ind(A100_1, A100_2))
print(stats.ttest_ind(A250_1, A250_2))
print(stats.ttest_ind(A500_1, A500_2))
print(stats.ttest_ind(A100_1, A250_1))
print(stats.ttest_ind(A100_1, A500_1))
print(stats.ttest_ind(A250_1, A500_1))
print(stats.ttest_ind(A100_2, A250_2))
print(stats.ttest_ind(A100_2, A500_2))
print(stats.ttest_ind(A250_2, A500_2))

print(stats.ttest_ind(D100_1, D100_2))
print(stats.ttest_ind(D250_1, D250_2))
print(stats.ttest_ind(D500_1, D500_2))
print(stats.ttest_ind(D100_1, D250_1))
print(stats.ttest_ind(D100_1, D500_1))
print(stats.ttest_ind(D250_1, D500_1))
print(stats.ttest_ind(D100_2, D250_2))
print(stats.ttest_ind(D100_2, D500_2))
print(stats.ttest_ind(D250_2, D500_2))

