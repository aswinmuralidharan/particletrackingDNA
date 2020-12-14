#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:43:03 2020

@author: aswinmuralidharan
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='STIXGeneral')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font', size=15)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=6)
plt.rc('mathtext', fontset='stix')

t = np.arange(0,10,0.001)
MSD = t**0.4
fig1, ax1 = plt.subplots(figsize=(5,5))
fig2, ax2 = plt.subplots(figsize=(5,5))
ax1.plot(t, MSD, 'r-', linewidth = 2, label = 'Subdiffusion')
ax2.loglog(t, MSD, 'r-', linewidth = 2, label = 'Subdiffusion')
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.tick_params(which='both', axis="both", labelbottom = False, labelleft = False)
ax2.set_ylim(1e-1,10)
ax2.set_xlim(1e-2,10)
ax2.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(which='both', axis="both", labelbottom = False, labelleft = False, direction="in")

ax1.set_ylim(0,10)
ax1.set_xlim(0,10)
ax1.set(ylabel=r'$\langle \Delta \bf{r}^2$ ($\mathrm{\tau}$)$\rangle$ ($\mathrm{\mu}$m$^2$)',
        xlabel=r'$\mathrm{\tau}$ (s)')
MSD = t**1
ax1.plot(t, MSD, 'k-', linewidth = 2, label = 'Brownian')
ax2.plot(t, MSD, 'k-', linewidth = 2, label = 'Brownian')
MSD = t**1.5
ax1.plot(t, MSD, 'b-', linewidth = 2, label = 'Superdiffusion')
ax2.plot(t, MSD, 'b-', linewidth = 2, label = 'Superdiffusion')
plt.tight_layout()
ax1.legend(frameon = True, edgecolor = 'black', loc = 'upper center', bbox_to_anchor = (0.5, 1.2), ncol = 3, fancybox = False, handletextpad=0.1, columnspacing = 0.1)
fig1.savefig('/Users/aswinmuralidharan/OneDrive/Progress_Meeting_Presentation/Media/scheme.eps')