#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate reference PSFs at different resolutions for cal.psffit.psffit.psffit.

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


def write_ref_psf_regular(outShape, subband_name):
    """
    Generate a coronagraphic image using a toy model of the DM.
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
    
    # Npad = 2048 #1236 #2472 #2048
    beam_diam_pix = 1000
    NpadFinal = int(2*np.ceil(ppl_desired*beam_diam_pix/2))  # even-valued
    ppl_true = NpadFinal/beam_diam_pix  # pixels per lambda/D
    padShapeFinal = (NpadFinal, NpadFinal)
    ppl_final = ppl_true*NpadFinal/outShape[0]

    # Load and zero-pad pupil masks
    fn_pupil = os.path.join(MASK_DIR, 'pupil', 'pupil_phaseC_1000_XCS.fits')
    pupil = fits.getdata(fn_pupil)

    # Propagation
    Efoc = fft2(inin(pupil, padShapeFinal))
    Ifoc = inin(np.abs(Efoc)**2, outShape) 
    Ifoc = Ifoc/np.max(Ifoc)
    
    hdu = fits.PrimaryHDU(Ifoc)
    hdu.writeto(os.path.join(HERE, 'ref_psf_band_%s.fits' % (subband_name)),
                overwrite=True)

    return Ifoc, ppl_final


def write_ref_psf_spc(outShape, subband_name, spc_name):
    """
    Generate a PSF of the SPM (no LS or FPM) using a toy model of the DM.
    
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
        fn_pupil = os.path.join(MASK_DIR, 'SPC_20200617_Spec', 'SPM_SPC-20200617_1000_rounded9_rotated_XCS.fits')
        beam_diam_pix = 1000
    elif spc_name.upper() == 'WFOV':
        fn_pupil = os.path.join(MASK_DIR, 'SPC_20200610_WFOV', 'SPM_SPC-20200610_1000_rounded9_gray_rotated_XCS.fits')
        beam_diam_pix = 1000
    else:
        raise ValueError('spc_name must be SPEC or WFOV')

    pupil = fits.getdata(fn_pupil)
    
    # Npad = 2048 #1236 #2472 #2048
    NpadFinal = int(2*np.ceil(ppl_desired*beam_diam_pix/2))  # even-valued
    ppl_true = NpadFinal/beam_diam_pix  # pixels per lambda/D
    padShapeFinal = (NpadFinal, NpadFinal)
    ppl_final = ppl_true*NpadFinal/outShape[0]

    # Propagation
    Efoc = fft2(inin(pupil, padShapeFinal))
    Ifoc = inin(np.abs(Efoc)**2, outShape) 
    Ifoc = Ifoc/np.max(Ifoc)
    
    hdu = fits.PrimaryHDU(Ifoc)
    hdu.writeto(os.path.join(HERE, 'ref_psf_band_%s_%s.fits' % (subband_name, spc_name.lower())),
                overwrite=True)

    return Ifoc, ppl_final


def write_files():
    
    out_shape = (200, 200)
    write_ref_psf_regular(out_shape, '1b')
    write_ref_psf_regular(out_shape, '2c')
    write_ref_psf_regular(out_shape, '3c')
    write_ref_psf_regular(out_shape, '4b')
    write_ref_psf_regular(out_shape, '4c')
    
    write_ref_psf_spc(out_shape, '2c', 'SPEC')
    write_ref_psf_spc(out_shape, '3c', 'SPEC')
    write_ref_psf_spc(out_shape, '1b', 'WFOV')
    write_ref_psf_spc(out_shape, '4b', 'WFOV')


if __name__ == '__main__':
    write_files()
