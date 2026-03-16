#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate downsampled SPC FPM representations.

NOTES on data types:
    Refer to https://note.nkmk.me/en/python-numpy-dtype-astype/
    i1 is uint8
    f4 is float32 (f for float, and 4 for 4 bytes)
    < = little-endian (LSB first)
    > = big-endian (MSB first)
"""
import os

from astropy.io import fits
import numpy as np

from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml

HERE = os.path.dirname(os.path.abspath(__file__))
AAC_DIR =  os.path.dirname(HERE)
FFT_DIR =  os.path.dirname(AAC_DIR)
MASK_DIR = os.path.join(AAC_DIR, 'mask_designs')

L1_SHAPE = (1200, 2200)
L2_SHAPE = (1024, 1024)

# subband wavelength table from cgisim_v3.0.7a.pdf
# maybe this belongs in a better place.
WAVELENGTH_LUT = loadyaml(os.path.join(AAC_DIR, 'misc',
                                       'subband_center_wavelengths.yaml'))

def fft2(arrayIn):
    """Perform an energy-conserving 2-D FFT including fftshift."""
    arrayOut = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayIn)))/np.sqrt(arrayIn.shape[0]*arrayIn.shape[1])

    return arrayOut


def write_excam_res_spc_fpm(outShape, subband_name, spc_name, fn_out=None):
    """
    Generate a downsampled SPC FPM representation.
    
    Parameters
    ----------
    outShape : array_like
        Desired array shape of the output image.
    subband_name : str
        Name of the Roman CGI subband filter to use.
    Returns
    -------
    img: array_like
        Normalized, 2-D coronagraphic image.
    ppl_true : float
        True number of pixels per lambda/D in the image.
    """
    subband_name = subband_name.lower()
    wavelength = WAVELENGTH_LUT[subband_name]
    wavelength0 = 500e-9  # meters
    ppl_desired = wavelength/wavelength0*2
    
    print('ppl_desired = %.3f for subband %s' % (ppl_desired, subband_name))
    
    # Load and zero-pad pupil masks
    if spc_name.upper() == 'SPEC':
        fn_fpm = os.path.join(MASK_DIR, 'SPC_20200617_Spec', 'FPM_SPC_20200617_Spec_25pixPerLamD.fits')
        ppl_ref = 25
    elif spc_name.upper() == 'WFOV':
        fn_fpm = os.path.join(MASK_DIR, 'SPC_20200610_WFOV', 'FPM_SPC_20200610_WFOV_25pixPerLamD.fits')
        ppl_ref = 25
    else:
        raise ValueError('spc_name must be SPEC or WFOV')

    fpm_ref = fits.getdata(fn_fpm)

    rotDeg = 90
    mag = ppl_desired/ppl_ref
    xOffset = 0
    yOffset = 0

    fpm_out = downsample(fpm_ref, rotDeg, mag, xOffset, yOffset, padFac=1.2, flipx=False)

    if fn_out is None:
        fn_out = os.path.join(HERE, 'fpm_%s_%s_excam_res.fits' % (subband_name, spc_name.lower()))
    fits.writeto(fn_out, fpm_out, overwrite=True)

    return fpm_out



def write_files():
    
    out_shape = (200, 200)

    write_excam_res_spc_fpm(out_shape, '3c', 'SPEC', fn_out=os.path.join(MASK_DIR, 'SPC_20200617_Spec', 'fpam_spc34_r6c1_excam_res.fits'))
    write_excam_res_spc_fpm(out_shape, '1b', 'WFOV', fn_out=os.path.join(MASK_DIR, 'SPC_20200610_WFOV', 'fpam_spc12_r2c1_excam_res.fits'))

if __name__ == '__main__':
    write_files()
