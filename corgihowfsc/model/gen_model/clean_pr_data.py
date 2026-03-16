"""
Load in all relevant phase retrieval data, clean it, and Fresnel propagate the E-field.
"""

import os
import sys
from datetime import datetime, timezone

from astropy.io import fits
import glob
import numpy as np
import skimage.morphology

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
SIT_PATH = os.path.join(HERE.split('cgi-sit')[0], 'cgi-sit')
UTIL_PATH = os.path.join(SIT_PATH, 'util')
MODEL_PATH = os.path.join(SIT_PATH, 'model')

# Calibration repo library
from cal.ampthresh.ampthresh import ampthresh
from cal.util import check
from cal.util.fresnelprop import fresnelprop
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util import remove_ptt
from cal.pr.util.unwrap import unwrap_segments


def clean_pr_cgisim(amp, ph_end2end, ph_backend, bandpass, pupil_backend=None):

    if bandpass == '1b':
        lam = 575e-9
    elif bandpass == '2c':
        lam = 660e-9
    elif bandpass in ('3c', '3d'):
        lam = 730e-9
    elif bandpass == '4b':
        lam = 825e-9
    else:
        raise ValueError(f'bandpass value {bandpass} not allowed here')

    # 1. Load and modify the backend phase map
    if ph_backend is not None:

        # Load back-end PR data
        lam_backend = 575e-9  # meters

        # Scale backend phase with wavelength because it was taken in Band 1B
        # only.
        ph_backend *= lam_backend/lam

        # Make back-end phase array the same size
        ph_backend = inin(ph_backend, ph_end2end.shape)

    else:
        ph_backend = np.zeros_like(ph_end2end)

    # 2. Subtract back-end phase from end-to-end phase (in E-field to avoid
    # issues with unwrapping making artifacts before Fresnel propagation)
    # to get front-end E-field at PIL.
    efield_pil = amp * np.exp(-1j*(ph_end2end - ph_backend))

    # 3. Remove piston/tip, and tilt from the front-end E-field at PIL plane.
    PARAM_PATH = os.path.join(HERE, 'out', 'pupil')
    fn_pr_prop_params = os.path.join(PARAM_PATH, f'params_read_and_prop_pr_band_{bandpass}.yaml')
    inputs = loadyaml(fn_pr_prop_params)
    ft_dir = inputs['ft_dir']
    diam_pupil_pix = inputs['diam_pupil_pix']
    ppl = inputs['ppl']
    fn_ref_psf = os.path.join(PARAM_PATH, inputs['fn_ref_psf'])
    fn_psffit_tuning = os.path.join(PARAM_PATH, inputs['fn_psffit_tuning'])
    nCrop = inputs['nCrop']
    nBin = inputs['nBin']
    nIter = inputs['nIter']
    min_size_zeros = inputs['min_size_zeros']

    phwrap_pupil, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
        amp,
        np.angle(efield_pil),
        ppl,
        diam_pupil_pix,
        ft_dir,
        fn_ref_psf,
        fn_psffit_tuning=fn_psffit_tuning,
        nCrop=nCrop,
        nBin=nBin,
        nIter=nIter,
    )
    amp_pupil = amp
    bmask_pupil = bmask

    # Remove isolated islands of zeroes in the software mask, which usually
    # happen from dead actuators where we want to keep the phase.
    bmask_pupil = np.logical_not(skimage.morphology.remove_small_objects(
        np.logical_not(bmask_pupil), min_size=min_size_zeros))

    # Unwrap the phase only after propagating back to the pupil to avoid
    # incorrectly masked spots from dead DM actuators.
    ph_pupil, _ = unwrap_segments(phwrap_pupil, amp_pupil, bMask=bmask_pupil, selem=None)


    if pupil_backend is None:
        pupil_backend = amp

    # Also remove PTT from the backend phase map
    phwrap_backend, bmask_backend, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
        pupil_backend,
        ph_backend,
        ppl,
        diam_pupil_pix,
        ft_dir,
        fn_ref_psf,
        fn_psffit_tuning=fn_psffit_tuning,
        nCrop=nCrop,
        nBin=nBin,
        nIter=nIter,
    )
    # Remove isolated islands of zeroes in the software mask, which usually
    # happen from dead actuators where we want to keep the phase.
    bmask_backend = np.logical_not(skimage.morphology.remove_small_objects(
        np.logical_not(bmask_backend), min_size=min_size_zeros))
    # Unwrap the phase only after propagating back to the pupil to avoid
    # incorrectly masked spots from dead DM actuators.
    ph_backend, _ = unwrap_segments(phwrap_backend, pupil_backend, bMask=bmask_backend, selem=None)

    return amp_pupil, ph_pupil, ph_backend, bmask_pupil, lam #, x_offset, y_offset


def OLD_clean_pr_cgisim(amp, phwrap, bandpass, subtract_backend=True):

    if bandpass == '1b':
        lam = 575e-9
    elif bandpass == '2c':
        lam = 660e-9
    elif bandpass == '3c':
        lam = 730e-9
    elif bandpass == '4b':
        lam = 825e-9
    else:
        raise ValueError(f'bandpass value {bandpass} not allowed here')

    # 1. Load and modify the backend phase map
    if subtract_backend:

        # Load back-end PR data
        lam_backend = 575e-9  # meters
        ph_backend = fits.getdata(os.path.join(HERE, 'data', 'pupil', '1b', 'poked_none', 'phase_backend.fits'))

        # Scale backend phase with wavelength because it was taken in Band 1B
        # only.
        ph_backend *= lam_backend/lam


        # Make back-end phase array the same size
        ph_backend = inin(ph_backend, phwrap.shape)

    else:
        ph_backend = np.zeros_like(phwrap)

    # 2. Subtract back-end phase from end-to-end phase (in E-field to avoid
    # issues with unwrapping making artifacts before Fresnel propagation)
    # to get front-end E-field at PIL.
    efield_pil = amp * np.exp(-1j*(phwrap - ph_backend))

    # 3. Remove piston/tip, and tilt from the front-end E-field at PIL plane.
    fn_pr_prop_params = os.path.join(HERE, 'data', 'pupil', f'params_read_and_prop_pr_band_{bandpass}.yaml')
    inputs = loadyaml(fn_pr_prop_params)
    ft_dir = inputs['ft_dir']
    diam_pupil_pix = inputs['diam_pupil_pix']
    ppl = inputs['ppl']
    fn_ref_psf = inputs['fn_ref_psf']
    fn_psffit_tuning = inputs['fn_psffit_tuning']
    nCrop = inputs['nCrop']
    nBin = inputs['nBin']
    nIter = inputs['nIter']
    min_size_zeros = inputs['min_size_zeros']

    ph_no_ptt, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
        amp,
        np.angle(efield_pil),
        ppl,
        diam_pupil_pix,
        ft_dir,
        fn_ref_psf,
        fn_psffit_tuning=fn_psffit_tuning,
        nCrop=nCrop,
        nBin=nBin,
        nIter=nIter,
    )

    phwrap_pupil = ph_no_ptt
    amp_pupil = amp
    bmask_pupil = bmask

    # # 4. Fresnel propagate from the PIL plane to the DM1 (i.e., pupil) plane.
    # # Fresnel propagate
    # # diam_pup_pix = 300
    # # diam_pup_m = 46.3e-3
    # pixpermeter = diam_pupil_pix/diam_pupil_m
    # if deltaz != 0:
    #     efield_pil = amp * np.exp(1j*ph_no_ptt)
    #     efield_pupil = fresnelprop(
    #         e=efield_pil,
    #         lam=lam,
    #         z=deltaz,
    #         nxfresnel=nxfresnel,
    #         pixpermeter=pixpermeter,
    #     )

    #     # 5. Convert E-field back to amplitude and unwrapped phase.
    #     phwrap_pupil = np.angle(efield_pupil)
    #     amp_pupil = np.abs(efield_pupil)
    #     bmask_pupil = ampthresh(pupilMap=amp, nBin=nBin)
    # else:
    #     phwrap_pupil = ph_no_ptt
    #     amp_pupil = amp
    #     bmask_pupil = bmask

    # Remove isolated islands of zeroes in the software mask, which usually
    # happen from dead actuators where we want to keep the phase.
    bmask_pupil = np.logical_not(skimage.morphology.remove_small_objects(
        np.logical_not(bmask_pupil), min_size=min_size_zeros))

    # Unwrap the phase only after propagating back to the pupil to avoid
    # incorrectly masked spots from dead DM actuators.
    ph_pupil, _ = unwrap_segments(
        phwrap_pupil, amp_pupil, bMask=bmask_pupil, selem=None)

    return amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam

