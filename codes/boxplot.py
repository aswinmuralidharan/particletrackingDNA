#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:54:25 2020

@author: aswinmuralidharan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')

fig1, ax1 = plt.subplots(figsize=(4,4))
ax1.boxplot([alpha_his100, alpha_his250, alpha_his500], showfliers = False, medianprops=dict(color='black'))
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which='both', axis="both", direction="in")
ax1.set_yscale('log')
ax1.set_ylim(1e-4,0.1)
plt.xticks([1, 2, 3], ['100 bp', '250 bp', '500 bp'])
ax1.set(ylabel=r'$D_{\mathrm{app}}$ ($\mathrm{\mu}$m$^2$/s$^\alpha$) ')
plt.tight_layout()
fig1.savefig(directory3 + '/Disbox.eps')