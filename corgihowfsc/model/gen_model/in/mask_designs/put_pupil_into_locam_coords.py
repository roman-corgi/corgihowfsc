#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For mask files used as references in LOCAM images, rotate/flip the design mask
representations so that they match the LOCAM coordinate system (LCS).

@author: ajriggs
"""
from astropy.io import fits
import numpy as np
import os

from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask

HERE = os.path.dirname(os.path.abspath(__file__))


# %% Input Pupil
folder_list = [
    'pupil',
]
fn_list = [
    'pupil_phaseC_1000.fits',
]

diam_in = 1000  # pixels
diam_out = 38  # pixels
narray_out = 50  # pixels

for ii in range(len(folder_list)):
    
    folder = folder_list[ii]
    fn_start = fn_list[ii]

    fn_new = ('pupil_phaseC_%d_LCS.fits' % 38)
    fn_in = os.path.join(HERE, folder, fn_start)
    fn_out = os.path.join(HERE, folder, fn_new)
    
    mask_in = fits.getdata(fn_in)
    mask_out = rotate_shift_downsample_amplitude_mask(mask_in, 270, diam_out/diam_in, 0, 0)
    # mask = np.rot90(mask, -1)  # Rotate 90 degrees CCW
    
    hdu = fits.PrimaryHDU(mask_out)
    hdu.writeto(fn_out, overwrite=True)
