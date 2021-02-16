# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:51:28 2019

@author: Aswin Muralidharan

"""
# For compatibility with Python 2 and 3
from __future__ import division, unicode_literals, print_function
import pims
import trackpy as tp
import os

Filepath = '/Volumes/Samsung_T5/Experimental Data/Hans'
bp = 'MCF10A500bp'  # Base pair to process
directory = Filepath + '/D_ROI_tiff/' + str(bp) # The input directory after background image corrections
tp.quiet() # Set the trackpy steps off so that the code runs faster
for filename in os.listdir(directory):
    """Import data series """
    print(filename)
    frames = pims.ImageSequence(os.path.join(directory, filename) + '/*.tif', as_grey=True)
    f = tp.batch(frames[0:350], 11, minmass=200, maxsize=4, noise_size=1, smoothing_size=15)

    """Link features into particle trajectories"""
    t = tp.link_df(f, 2, memory=3)
    """ max displacement 2 pixels
    # memory missed particle is 3"""

    tf = tp.filter_stubs(t, 85)
    """Filter out spurious tracks minimum of 50 frames"""

    d = tp.compute_drift(tf)
    tm = tp.subtract_drift(tf.copy(), d)
    """Remove the overall drift"""
    
    for dirs in ['/f', '/t', '/tf', '/im', '/im_plot', '/trajectories', '/tm']:
        if not os.path.exists(Filepath + '/E_output_data/' + str(bp) + dirs):
            os.makedirs(Filepath + '/E_output_data/' + str(bp) + dirs)
    """Store Data"""
    f.to_csv(Filepath + '/E_output_data/' + str(bp) + '/f/f_' + filename + '.csv')
    t.to_csv(Filepath + '/E_output_data/' + str(bp) + '/t/t _' + filename + '.csv')
    tf.to_csv(Filepath + '/E_output_data/' + str(bp) + '/tf/tf_' + filename + '.csv')
    tm.to_csv(Filepath + '/E_output_data/' + str(bp) + '/tm/tm_' + filename + '.csv')

