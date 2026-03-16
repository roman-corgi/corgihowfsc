#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For mask files used as references in EXCAM images, rotate/flip the design mask
representations so that they match the EXCAM coordinate system (XCS).

@author: ajriggs
"""
from astropy.io import fits
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

from gsw.cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample
from gsw.cal.util.insertinto import insertinto as inin

# dorba and import cal tools
import ctc.dorba
ctc.dorba.init(data_subdir=os.environ['USER'])

# %% Shaped Pupil Masks
folder_list = [
    'SPC_20200617_Spec',
    'SPC_20200610_WFOV',
    'SPC_20200628_Spec_Rot60',
]
fn_list = [
    'SPM_SPC-20200617_1000_rounded9_rotated.fits',
    'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits',
]

for ii in range(len(folder_list)):

    folder = folder_list[ii]
    fn_start = fn_list[ii]

    fn_new = fn_start[:-5] + '_XCS' + fn_start[-5::]
    fn_in = os.path.join(HERE, folder, fn_start)
    fn_out = os.path.join(HERE, folder, fn_new)

    mask = fits.getdata(fn_in)
    mask = np.rot90(mask, -3)  # Rotate 270 degrees CCW

    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(fn_out, overwrite=True)

# SPECROT:
folder = 'SPC_20200628_Spec_Rot60'
fn_start = 'SPM_SPC_20200628_Spec_Rot60_1000us3x_binary_uint8.fits'
fn_new = 'SPM_SPC_20200628_Spec_Rot60_1000' + '_XCS' + fn_start[-5::]
fn_in = os.path.join(HERE, folder, fn_start)
fn_out = os.path.join(HERE, folder, fn_new)

mask = fits.getdata(fn_in).astype(float)
mask = downsample(maskIn=mask,
                  rotDeg=30.0,
                  mag=1/3,
                  xOffset=0,
                  yOffset=0,
                  padFac=1.2,
                  flipx=False,
                  )
mask = inin(arr0=mask, outsize=(1001, 1001))
hdu = fits.PrimaryHDU(mask)
hdu.writeto(fn_out, overwrite=True)


# # %% Lyot Stops
# folder_list = [
#     'HLC_20190210b_NFOV',
#     'SPC_20200617_Spec',
#     'SPC_20200610_WFOV',
# ]
# fn_list = [
#     'LS_HLC_20190210b_NFOV_1000.fits',
#     'LS_SPC_20200617_Spec_1000.fits',
#     'LS_SPC_20200610_WFOV_1000.fits',
# ]

# for ii in range(len(folder_list)):
    
#     folder = folder_list[ii]
#     fn_start = fn_list[ii]

#     fn_new = fn_start[:-5] + '_XCS' + fn_start[-5::]
#     fn_in = os.path.join(HERE, folder, fn_start)
#     fn_out = os.path.join(HERE, folder, fn_new)
    
#     mask = fits.getdata(fn_in)
#     mask = np.rot90(mask, -3)  # Rotate 270 degrees CCW
    
#     hdu = fits.PrimaryHDU(mask)
#     hdu.writeto(fn_out, overwrite=True)


# # %% Input Pupil
# folder_list = [
#     'pupil',
# ]
# fn_list = [
#     'pupil_phaseC_1000.fits',
# ]

# for ii in range(len(folder_list)):
    
#     folder = folder_list[ii]
#     fn_start = fn_list[ii]

#     fn_new = fn_start[:-5] + '_XCS' + fn_start[-5::]
#     fn_in = os.path.join(HERE, folder, fn_start)
#     fn_out = os.path.join(HERE, folder, fn_new)
    
#     mask = fits.getdata(fn_in)
#     mask = np.rot90(mask, -1)  # Rotate 90 degrees CCW
    
#     hdu = fits.PrimaryHDU(mask)
#     hdu.writeto(fn_out, overwrite=True)
