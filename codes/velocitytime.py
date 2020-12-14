#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:30:52 2020

@author: aswinmuralidharan
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import string
#from matplotlib.ticker import FuncFormatter, MultipleLocator

plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=10)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')

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

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = '100bp'  # Base pair to process
bps = '100 bp'
for dirs in ['/MSDcollected', '/Trajcollected','/MSDindividual', '/Figures' ]:
    if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
        os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
directory = Filepath + '/E_output_data/' + str(bp) + '/tm'
directory2 = Filepath + '/E_output_data/' + str(bp) + '/MSDindividual'
directory3 = Filepath + '/E_output_data/' + str(bp) + '/Figures'
mpp=0.16
lagtime = 1
max_lagtime=100
video = 1
#for filename in os.listdir(directory):
#if not filename.startswith('.'):
traj = pd.read_csv(directory+'/'+ 'tm_movie_7.csv')
pos_columns = ['x', 'y']
vel = []
for pid, ptraj in traj.groupby('particle'):
    pos = ptraj.set_index('frame')[pos_columns] * mpp/0.001
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    pos = trajectory(pos, pid, max_lagtime,video, pos_columns)
    disp = pos.sub(pos.shift(lagtime))
    disp = (disp['x']**2+disp['y']**2)**0.5
    if disp.max() >200:
        fig1,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(disp, alpha = 0.5)
        ax2.plot(pos['x'],pos['y'])