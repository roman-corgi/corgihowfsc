# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compute the true ExCAM plate scale using the PSF of a SPM.

This was tested using the following package versions:
- roman_phasec_jpl_v1.3
- cgisim_jpl_v3.0.8
- emccd_detect-2.2.0
"""
import os
import pathlib
import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pylab as plt

import cgisim as cgisim
import proper

import cal.util.check as check
from cal.util.findopt import find_optimum_1d
from cal.util.insertinto import insertinto
from cal.util.loadyaml import loadyaml
import cal.maskgen.maskgen as mg

MASKGEN_PATH = pathlib.Path(mg.__file__).resolve().parent


HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def run_calib():
    """Compute the true ExCAM plate scale with the PSF of a SPM."""
    imshape = (201, 201)

    plateScaleList = []  # initialize
    wavelengthList = []  # initialize
    bandpassList = ['1a', '1b', '1c',
                    '2a', '2b', '2c',
                    '3a', '3b', '3c', '3d', '3e',
                    '4a', '4b', '4c'
                    ]

    v_dm1 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm2_v.fits' ))

    for bandpass in bandpassList:

        if '1' in bandpass:
            cor_type = 'spc-wide_band1'
        elif '2' in bandpass:
            cor_type = 'spc-spec_band2'
        elif '3' in bandpass:
            cor_type = 'spc-spec_band3'
        elif '4' in bandpass:
            cor_type = 'spc-wide_band4'

        if bandpass == '1':
            lam0_um = 0.575
        elif bandpass == '1a':
            lam0_um = 0.556
        elif bandpass == '1b':
            lam0_um = 0.5749995
        elif bandpass == '1c':
            lam0_um = 0.594

        elif bandpass == '2':
            lam0_um = 0.66
        elif bandpass == '2a':
            lam0_um = 0.615
        elif bandpass == '2b':
            lam0_um = 0.638
        elif bandpass == '2c':
            lam0_um = 0.656

        elif bandpass == '3':
            lam0_um = 0.73
        elif bandpass == '3a':
            lam0_um = 0.681
        elif bandpass == '3b':
            lam0_um = 0.704
        elif bandpass == '3c':
            lam0_um = 0.727
        elif bandpass == '3d':
            lam0_um = 0.754
        elif bandpass == '3e':
            lam0_um = 0.7775

        elif bandpass == '4':
            lam0_um = 0.825
        elif bandpass == '4a':
            lam0_um = 0.792
        elif bandpass == '4b':
            lam0_um = 0.825
        elif bandpass == '4c':
            lam0_um = 0.857

        nlam = 1
        lam_array = lam0_um

        nout = 512
        final_sampling_m = 13e-6

        nspm = 1000
        fn_spm = os.path.join(
            MASKGEN_PATH, 'maskdesigns', 'SPC_20200617_Spec',
            'SPM_SPC-20200617_1000_rounded9_rotated.fits'
        )
        spm = fits.getdata(fn_spm)

        params = {
            'pupil_mask_array': spm,
            'ccd':{},
            # 'use_pupil_lens':1,
            'use_errors':1,
            # 'polaxis':polaxis, #np.abs(polaxis),
            'use_fpm':0,
            'use_lyot_stop':0,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }



        # Use cgisim
        cgi_mode = 'excam' #'excam_efield'
        polaxis = -10
        # star_spectrum = 'a0v'
        # star_vmag = 2.0
        # ccd_fn = os.path.join(IN_PATH, 'ccd_params_no_noise.yaml')
        # ccd = loadyaml(ccd_fn)
        # ccd.update({'exptime': 0.1})
        print("Computing image")
        psf, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd={},
        )
        image0 = insertinto(psf, imshape)
        # field_cube, _ = cgisim.rcgisim(
        #     cgi_mode, cor_type, bandpass, polaxis, params, ccd={},
        # )
        # field = np.mean(field_cube, axis=0)
        # field = insertinto(field, imshape)
        # amp = np.abs(field)
        # image0 = amp**2
        image = image0/np.max(image0)


        plt.figure(1)
        plt.clf()
        plt.imshow(np.log10(image))
        plt.gca().invert_yaxis()
        plt.clim(0, -6)
        plt.colorbar()
        plt.pause(1e-2)

        # %% Compute Reference PSF with MFT

        fl = 1
        D = 1
        dx = D / nspm

        ppl0 = lam0_um/0.500*2*0.99
        pplEst = ppl0

        nMag = 31
        magVec = 1 + np.linspace(-2, 2, nMag)/100
        nIter = 3

        thresh = 1e-3

        for iter_ in range(nIter):
            costVec = np.zeros(nMag)
            for iMag, mag in enumerate(magVec):

                ppl = pplEst * mag
                dxi = 1/ppl
                deta = dxi
                E = mft_p2f(np.rot90(spm, -1), fl, 1, dx, dxi, nout, deta, nout)
                im = np.abs(E)**2
                im = im/np.max(im)
                im = insertinto(im, imshape)
                diff = np.abs(image - im)

                imageNew = image.copy()
                imageNew[imageNew > thresh] = thresh
                imNew = im
                imNew[imNew > thresh] = thresh
                diff = np.abs(imageNew - imNew)

                costVec[iMag] = np.sum(diff)

                # plt.figure(2)
                # plt.clf()
                # plt.imshow(np.log10(diff))
                # plt.gca().invert_yaxis()
                # plt.colorbar()
                # plt.pause(1e-2)

                # plt.figure(3)
                # plt.clf()
                # plt.imshow(np.log10(im))
                # plt.gca().invert_yaxis()
                # plt.colorbar()
                # plt.pause(1e-2)

            tempVec = (costVec - np.min(costVec))**2

            plt.figure(4)
            plt.clf()
            plt.plot(magVec, tempVec)
            plt.pause(1)

            # plt.show()

            # # Compute reference image for best estimate
            # dxi = 1/pplEst
            # deta = dxi
            # E = mft_p2f(spm, fl, 1, dx, dxi, nout, deta, nout)
            # im = np.abs(E)**2
            # im = insertinto(im, imshape)
            # imBest = im/np.max(im)
            # imNew = imBest
            # imNew[imNew > thresh] = thresh
            # diffBest = np.abs(imageNew - imNew)

            # plt.figure()
            # plt.imshow(np.log10(imageNew))
            # plt.gca().invert_yaxis()
            # plt.clim(0, -6)
            # plt.colorbar()
            # plt.pause(1e-2)

            # plt.figure()
            # plt.imshow(np.log10(imNew))
            # plt.gca().invert_yaxis()
            # plt.clim(0, -6)
            # plt.colorbar()
            # plt.pause(1e-2)

            # plt.figure()
            # plt.imshow(np.log10(diffBest+1e-12))
            # plt.gca().invert_yaxis()
            # plt.clim(0, -6)
            # plt.colorbar()
            # plt.pause(1e-2)

            # plt.figure()
            # plt.plot(magVec*pplEst, tempVec)
            # plt.pause(1e-2)


            if iter_ == 0:
                bestInd = np.argmin(costVec)
                pplEst *= magVec[bestInd]
            else:
                magBest = find_optimum_1d(magVec, tempVec)
                pplEst *= magBest

        print('ppl = %.4f in Band %s' % (pplEst, bandpass))

        plateScaleList.append(pplEst)
        wavelengthList.append(lam0_um)

    data = {
        'wavelength (um)': wavelengthList,
        'bandpass': bandpassList,
        'true plate scale (ppl)': plateScaleList,
    }

    df = pd.DataFrame(data=data)
    # row_labels = bandpassList
    # df = pd.DataFrame(data=data, index=row_labels)

    out_name = os.path.join(OUT_PATH, 'focal_plane', 'true_plate_scale_values.csv')
    df.to_csv(out_name)

    plt.figure(1)
    plt.plot(wavelengthList, plateScaleList)
    plt.pause(1e-1)


def mft_p2f(E_pup, fl, wavelength, dx, dxi, Nxi, deta, Neta,
            centering='pixel'):
    """
    Propagate a pupil to a focus using a matrix-multiply DFT.

    Parameters
    ----------
    E_pup : array_like
        Electric field array in pupil plane
    fl : float
        Focal length of Fourier transforming lens
    wavelength : float
        Propagation wavelength
    dx : float
        Step size along either axis of focal plane.  The vertical and
        horizontal step sizes are assumed to be equal.
    dxi : float
        Step size along horizontal axis of focal plane
    Nxi : int
        Number of samples along horizontal axis of focal plane.
    deta : float
        Step size along vertical axis of focal plane
    Neta : int
        Number of samples along vertical axis of focal plane.
    centering : string
        Whether the input and output arrays are pixel-centered or
         interpixel-centered. Possible values: 'pixel', 'interpixel'

    Returns
    -------
    array_like
        Field in pupil plane, after propagating through a lens.

    """
    # if centering not in _VALID_CENTERING:
    #     raise ValueError(_CENTERING_ERR)
    check.twoD_array(E_pup, 'E_pup', TypeError)
    check.real_scalar(fl, 'fl', TypeError)
    check.real_positive_scalar(wavelength, 'wavelength', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_positive_scalar(dxi, 'dxi', TypeError)
    check.positive_scalar_integer(Nxi, 'Nxi', TypeError)
    check.real_positive_scalar(deta, 'deta', TypeError)
    check.positive_scalar_integer(Neta, 'Neta', TypeError)

    dy = dx
    M, N = E_pup.shape
    if M != N:
        raise ValueError('Input array is not square')

    # Pupil-plane coordinates
    # Broadcast to column vector
    x = create_axis(N, dx, centering=centering)[:, None]
    y = x.T  # Row vector

    # Focal-plane coordinates
    # Broadcast to row vector
    xi = create_axis(Nxi, dxi, centering=centering)[None, :]
    # Broadcast to column vector
    eta = create_axis(Neta, deta, centering=centering)[:, None]

    # Fourier transform matrices
    pre = np.exp(-2 * np.pi * 1j * (eta * y) / (wavelength * fl))
    post = np.exp(-2 * np.pi * 1j * (x * xi) / (wavelength * fl))

    # Constant scaling factor in front of Fourier transform
    scaling = np.sqrt(dx * dy * dxi * deta) / (1 * wavelength * fl)

    return scaling * np.linalg.multi_dot([pre, E_pup, post])


def create_axis(N, step, centering='pixel'):
    """
    Create a one-dimensional coordinate axis with a given size and step size.

    Can be constructed to follow either the FFT (pixel-centered) interpixel-
    centered convention, which differ by half a pixel for even-sized arrays.
    For odd-sized arrays, both values of centering put the center on the center
    pixel.

    Parameters
    ----------
    N : int
        Number of pixels in output axis
    step : float
        Physical step size between axis elements
    centering : 'pixel' or 'interpixel'
        Centering of the coordinates in the array.  Note that if N is odd, the
        result will be pixel-centered regardless of the value of this keyword.

    Returns
    -------
    array_like
        The output coordinate axis
    """
    check.positive_scalar_integer(N, 'N', TypeError)
    check.real_positive_scalar(step, 'step', TypeError)
    # check.centering(centering)

    axis = np.arange(-N // 2, N // 2, dtype=np.float64) * step
    even = not N % 2  # Even number of samples?

    if even and (centering == 'interpixel'):
        # Inter-pixel-centering onlyif the number of samples is even
        axis += 0.5 * step

    return axis


if __name__ == '__main__':
    run_calib()
