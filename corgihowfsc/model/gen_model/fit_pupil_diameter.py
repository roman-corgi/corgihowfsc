# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test accuracy of SPAM calibration."""
import os
import pathlib
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import pandas as pd

import cgisim as cgisim
import proper

from cal.util.loadyaml import loadyaml
from cal.util.insertinto import insertinto
from cal.pupilfit import pupilfit_open as pfo

PUPILFIT_PATH = pathlib.Path(pfo.__file__).resolve().parent

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def fit_pupil_diameter():
    """Get the pupil diameter for each subband filter."""
    flagPlot = True

    cgi_mode = 'excam' #_efield'
    imshape = (351, 351)
    polaxis = -10  # compute images for mean X+Y polarization
    # use_errors = True
    # star_spectrum = 'a0v'  # 'k5v'
    # star_vmag = 2.0

    v_dm1 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm2_v.fits' ))

    # ccd_fn = os.path.join(IN_PATH, 'ccd_params_for_pupil_imaging.yaml')
    # ccd = loadyaml(ccd_fn)
    # ccd.update({'exptime': 0.8})
    ccd = {}

    # magVec = [1, ]  # np.linspace(0.95, 1.05, 6)
    # rotVec = [0, ]  # np.arange(-2, 3)

    bandList = ['1b', '2c', '3c', '4b']
    rotList = []
    diamList = []
    xcList = []
    ycList = []

    # magTrue = 1
    # rotTrue = 0

    for bandpass in bandList:

        if bandpass == '1b':
            cor_type = 'hlc_band1'
        elif bandpass == '2c':
            cor_type = 'hlc_band2'
        elif bandpass == '3c':
            cor_type = 'hlc_band3'
        elif bandpass == '4b':
            cor_type = 'hlc_band4'
        else:
            raise ValueError(f'bandpass value {bandpass} not accepted here.')

        # # Store lists of true values
        # rotTrueList.append(rotTrue)
        # magTrueList.append(magTrue)

        # pupil_mask_fn = os.path.join(
        #     OUT_PATH,
        #     'spam',
        #     'spm_calib_spots_n309_mag%.3f_clk%.3fdeg.fits' % (magTrue, rotTrue)
        # )
        # pupil_mask_array = fits.getdata(pupil_mask_fn)

        # nActs = 48
        # dm1 = np.zeros((nActs, nActs))
        # dm2 = np.zeros((nActs, nActs))

        print("Computing pupil image")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            'ccd':{},
            'use_pupil_lens':1,
            'use_errors':1,
            'use_fpm':0,
            'use_lyot_stop':0,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        img, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        # field = np.mean(field_cube, axis=0)
        image0 = insertinto(img, imshape)
        image = image0/np.max(image0)
        # image -= ccd['bias']
        # image = np.rot90(image, -1)  # 90 deg CCW rot to be in EXCAM coords
        # image = insertinto(image, (gridsize, gridsize))
        # image = np.roll(image, (yOffset, xOffset), axis=(0, 1))
        # image[image < 0] = 0

        if flagPlot:
            plt.figure(1)
            plt.clf()
            plt.title(f'Band {bandpass.upper()}')
            plt.imshow(image)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.pause(1)

            # plt.show()


        # ######################################################
        # Fit the unmasked pupil parameters
        fn_tuning = os.path.join(IN_PATH, 'any_band', 'params_fit_unmasked_pupil.yaml')
        xOffset, yOffset, clockEst, diamEst = pfo.fit_unmasked_pupil(image, fn_tuning, data_path=DATA_PATH_PUPIL)
        xcList.append(xOffset)
        ycList.append(yOffset)
        rotList.append(clockEst)
        diamList.append(diamEst)

        print('Band = %s  diam = %.4f\n' % (bandpass, diamEst))

        # magEst, rotEst = pupilfit_gsw.fit_shaped_pupil_mag_clocking(image, fnYAML)
        # print('rotEst = %.3f  magEst = %.4f\n' % (rotEst, magEst))

        # tuning_dict = loadyaml(fnYAML)
        # nBeamNom = tuning_dict['nBeamNom']  # TODO
        # nBeam = nBeamNom*magEst
        # diamList.append(nBeam)

    data = {
        'band': bandList,
        'beamDiamPix': diamList,
        'clocking (degrees)': rotList,
        'xoffset': xcList,
        'yoffset': ycList,
    }

    df = pd.DataFrame(data=data)  # , index=row_labels)

    out_name = os.path.join(OUT_PATH, 'pupil', 'pupil_fitting_results.csv')
    df.to_csv(out_name)


if __name__ == '__main__':
    fit_pupil_diameter()
