#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:25:12 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
import trackpy as tp
from matplotlib.ticker import MaxNLocator

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

def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.18, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.18, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
        

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bps = ['100bp', '250bp','500bp']
bpcs = ['100 bp', '250 bp','500 bp']
mpp = 0.16
fps = 10
max_lagtime = 100
pos_columns = ['x', 'y']
trajec = []
fig1 = plt.figure(figsize=(3.375*2,3.375*2*1.6/3))

ax1 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan =1)
ax2 = plt.subplot2grid((2, 3), (0, 0) )
ax3 = plt.subplot2grid((2, 3), (0, 1))
ax4 = plt.subplot2grid((2, 3), (1, 0))
ax5 = plt.subplot2grid((2, 3), (1, 1))
fig2 = plt.figure(figsize=(3.375*2,3.375*2*1.6/3))
ax6 = plt.subplot2grid((2, 3), (1, 1),rowspan=2, colspan =2)
num_plots1 = 1
num_plots2 = 1
axs = (ax3,ax4,ax5)
fig3, ((ax7),(ax8)) = plt.subplots(2,1,figsize=(3.375,3.375*0.4*2),sharex=True)

axnum=0
for bp in bps:
    video = 1
    dispfromoriginmax = []
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
        x = traj['x'].head(201)
        y = traj['y'].head(201)
        dispfromoriginmax.append(np.sqrt(x**2+y**2).max())
        disp = (x.sub(x.shift(1))**2+y.sub(y.shift(1))**2)**0.5
        dispmax = ((x[0]-x[x.index[-1]])**2+(y[0]-y[y.index[-1]])**2)**0.5
        if dispmax.max() >1 and np.size(x)>150:
            if num_plots1 ==4:
                t = np.arange(0, 20.1, 0.1)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(t.min(), t.max())
                lc = LineCollection(segments, cmap='plasma', norm=norm, zorder =2, alpha =1)
                lc.set_array(t)
                lc.set_linewidth(1.5)
                line = ax1.add_collection(lc)
                ax1.plot(x,y, marker ='o', markerfacecolor = 'lightgray', markeredgecolor = 'lightgray', linestyle = 'None', zorder=1)
                deltat = np.arange(0,np.size(x.sub(x.shift(1))))/10
                ax7.plot(deltat, x.sub(x.shift(1)),color = 'blue')
                ax7.plot(deltat, y.sub(y.shift(1)),color = 'red')
                ax8.plot(deltat, disp,color = 'black')
            num_plots1 += 1
        if dispmax.max()<0.1 and np.size(x)>200:
            if  num_plots2 == 5:
                x = [element-0.4 for element in x]
                y = [element-0.1 for element in y]
                t = np.arange(0, 20.1, 0.1)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(t.min(), t.max())
                lc = LineCollection(segments, cmap='plasma', norm=norm, zorder =2, alpha =1)
                lc.set_array(t)
                lc.set_linewidth(1.5)
                line = ax1.add_collection(lc)
                ax1.plot(x,y, marker ='o', markerfacecolor = '#91bfdb', markeredgecolor = '#91bfdb', linestyle = 'None', zorder=1)
            num_plots2+=1
    axs[axnum].hist(dispfromoriginmax,bins = np.arange(0,2,0.1), edgecolor='black',color = 'white', density = True, rwidth=1)
    axs[axnum].set(xlabel=r'$ \left |\Updelta \mathbf{r}_{\mathrm{max}}\right |$' +' '+ r'(\textmu m) ',
        ylabel=r'$P(\left |\Updelta \mathbf{r}_{\mathrm{max}}\right |)$'+' '+ r'(\textmu m$^{-1}$) ')
    axs[axnum].yaxis.set_ticks_position('both')
    axs[axnum].xaxis.set_ticks_position('both')
    axs[axnum].tick_params(which='both', axis="both", direction="in")
    axs[axnum].set_xlim((-0.1, 2))
    axs[axnum].set_ylim((0, 2.7))
    axs[axnum].text(0.95, 0.95, bpcs[axnum], verticalalignment='top', horizontalalignment='right',
         multialignment="left",
         transform=axs[axnum].transAxes, color = 'black')
    axnum+=1
                
   
scalebary = [-1.4, -1.4]
scalebarx = [-0.1,0.15]
ax1.plot(scalebarx,scalebary,'k', linewidth = 2)
ax1.set_xlim(-0.7, 0.2)
ax1.set_ylim(-1.5,0.3) 
ax1.text(-0.45,-0.2, r'\textit{Localized}', verticalalignment='top', horizontalalignment='center', style ='italic' ) 
ax1.text(-0.05,0.1, r'\textit{Directional}', verticalalignment='top', horizontalalignment='center', style ='italic' ) 
ax1.text(-0.5,-1.32, r'\textit{Turn}', verticalalignment='top', horizontalalignment='center', style ='italic' )
ax1.text(0.025,-1.35, r'\textbf{250 nm}', verticalalignment='bottom', horizontalalignment='center' )             
cbaxes = inset_axes(ax1, width="40%", height="3%", loc=2) 
cbar = fig1.colorbar(line, cax=cbaxes, label = '$t$ (s)', orientation='horizontal')
cbar.ax.tick_params(axis='x', direction='in')
cbar.ax.get_xaxis().labelpad = -7
cbar.set_ticks(np.arange(0, 20.1, step=20))
ax1.set_aspect(1)
for n, ax in enumerate((ax2, ax3, ax4, ax5, ax1)):   
    ax.text(-0.2, 1, r'\textbf{'+ string.ascii_lowercase[n]+'}', transform=ax.transAxes, 
            size=8)
im = mpimg.imread('/Volumes/Samsung_T5/Experimental Data/Hans/B_raw_tiff/Schema.png')
image1 = ax6.imshow(im, zorder =1)
ax1.axes.xaxis.set_visible(False)
ax1.axes.yaxis.set_visible(False)
ax6.axes.xaxis.set_visible(False)
ax6.axes.yaxis.set_visible(False)
tf = pd.read_csv('/Volumes/Samsung_T5/Experimental Data/Hans/E_output_data/100bp/tf/tf_movie_9.csv')
tp.plot_traj(tf,label = False, ax = ax6, zorder = 2, alpha = 0.5)
fig1.tight_layout() 
fig1.savefig(directory3 + '/scheme.pdf', dpi = 300)
fig2.savefig(directory3 + '/scheme1.png', dpi = 300)
ax7.set_xlim(0, 14)
ax7.set_ylim(-0.1,0.1)
ax8.set_xlim(0, 14)
ax8.set_ylim(0,0.15)
ax8.set(xlabel=r'$ t$' +' '+ r'(s) ',
        ylabel=r'$\left |\Updelta \mathbf{r}\right |$'+' '+ r'(\textmu m) ')
#ax7.set(ylabel=r'$\Updelta x$'+' '+ r'(\textmu m) ')
multicolor_ylabel(ax7,(r'$\Updelta x$', r'$\Updelta y$',' '+ r'(\textmu m) '),('k','r','b'),axis='y')
for n, ax in enumerate((ax7, ax8)):   
    ax.text(-0.2, 1, r'\textbf{'+ string.ascii_lowercase[n]+'}', transform=ax.transAxes, 
            size=8)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', axis="both", direction="in")
fig3.tight_layout()
fig3.savefig(directory3 + '/sample.pdf', dpi = 300)
#frames = pims.ImageSequence('../sample_data/bulk_water/*.png', as_grey=True)       