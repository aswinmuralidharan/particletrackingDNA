# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:19:50 2019

@author: amuralidharan
"""
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import pims
import os
import numpy as np
from scipy.ndimage.filters import uniform_filter1d, correlate1d
from PIL import Image

def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def gaussian_kernel(sigma, truncate = 4.0):
    "1D discretized gaussian"
    lw = int(truncate * sigma + 0.5)      #calculate the length of the gaussian_kernel
    x = np.arange(-lw, lw+1)              #convert into a list
    result = np.exp(x**2/(-2*sigma**2))   #calculate the gaussian curve
    return result / np.sum(result)        #normalize by the summation


"""Gaussian filter"""
def lowpass(image, sigma=1, truncate=4):
    sigma = validate_tuple(sigma, image.ndim) #convert sigma to 2 dimension
    result = np.array(image, dtype=np.float)  #convert image to np.float
    for axis, _sigma in enumerate(sigma):     #convolve gaussian about the x and y axis 
        if _sigma > 0:
            correlate1d(result, gaussian_kernel(_sigma, truncate), axis,
                        output=result, mode='constant', cval=0.0)
    return result


"""Boxcar average"""
def boxcar(image, size):
    size = validate_tuple(size, image.ndim)  #convert size in 2 dimension
    if not np.all([x & 1 for x in size]):    #check if size is odd integer
        raise ValueError("Smoothing size must be an odd integer. Round up.")
    result = image.copy()                    #store image as results
    for axis, _size in enumerate(size):      #boxcar average the image about the x and y axis 
        if _size > 1:
            uniform_filter1d(result, _size, axis, output=result,
                             mode='nearest', cval=0)
    return result


def bandpass(image, lshort, llong, threshold=None, truncate=4):
    lshort = validate_tuple(lshort, image.ndim)             #size of the gaussian kernel
    llong = validate_tuple(llong, image.ndim)               #size of the boxcar average
    if np.any([x >= y for (x, y) in zip(lshort, llong)]): 
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):          #Clip bandpass for integer type images
            threshold = 1
        else:
            threshold = 1/250.                              #Clip bandpass for float images 
    gauss      = lowpass(image, lshort, truncate)           #Perfrom gaussian filter
    background = boxcar(image, llong)                       #Calculate background
    result1    = gauss - background                         #Substract background
    result2    = np.where(result1 >= threshold, result1, 0) #Apply bandpass filter 
    return result2, result1, background, gauss              #Return processed image, gaussian fit, and background


bp = 'MCF10A500bp' # Enter the folder name
N = 350 # Number of frames 
filepath = '/Volumes/Samsung_T5/Experimental Data/Hans/'
directory = filepath + '/B_raw_tiff/' + str(bp) + '/'
directory2 = filepath + 'C_processed_tiff/' + str(bp) +'/'
# Loop through the directory
for filename in os.listdir(directory):
    if not filename.startswith('.'):
        # Open the image sequence from the directory as tif images. 
        frames   = pims.ImageSequence(directory + filename +'/*.tif', as_grey=True)
        # If the folders do not exist in the directory, create them
        for dirs in [filename]:
            if not os.path.exists(directory2 + dirs):
                os.makedirs(directory2 + dirs)
        print('processing video' + filename)
        # Apply the bandpass filter
        for i in range(N):
                image = frames[i]
                a, b, c, d = bandpass(image,1,15, threshold = 40)
                im = Image.fromarray(a)
                im.save(directory2 + filename+ '/mov_' + str(1) +'_{0}.tiff'.format(i))

