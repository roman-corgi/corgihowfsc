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

from cal.pupilfit import pupilfit_gsw
from cal.util.insertinto import insertinto
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml

PUPILFIT_PATH = pathlib.Path(pupilfit_gsw.__file__).resolve().parent

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def create_tuning_yaml():
    #bandList = ['1b', '2c', '3c', '4b']
    fn_tuning_0 = os.path.join(IN_PATH, 'any_band', 'params_fit_spam_mag_clocking_template.yaml')
    dict0 = loadyaml(fn_tuning_0)

    diam_df = pd.read_csv(os.path.join(OUT_PATH, 'pupil', 'pupil_fitting_results.csv'))
    df = diam_df[['band', 'beamDiamPix']]
    # df.loc[df['band'] == '2c', 'beamDiamPix']

    bandList = list(diam_df.values[:, 1])
    diamList = diam_df.values[:, 2]

    for index, bandpass in enumerate(bandList):

        # Access a specific column value based on a condition
        # This returns a Series of values that meet the criteria
        # diam_obj = diam_df.loc[diam_df['band'] == bandpass, 'beamDiamPix']
        # print(diam_obj)
        # diam = float(diam_obj[0])
        diam = diamList[index]
        print(f'Writing the beam diameter {diam} for band {bandpass}')
        dictNew = dict0.copy()
        dictNew['nBeamNom'] = float(diam)

        fn_tuning_new = os.path.join(OUT_PATH, 'spam', 'params_fit_spam_mag_clocking_band%s.yaml' % bandpass.lower())
        writeyaml(dictNew, fn_tuning_new)

    return None


def calibrate_spam():
    """Test accuracy of fit_shaped_pupil_mag_clocking() over full ranges."""
    flagPlot = True

    cgi_mode = 'excam_efield'
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
    rotTrueList = []
    magTrueList = []
    rotEstList = []
    magEstList = []
    diamList = []

    magTrue = 1
    rotTrue = 0

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

        # Store lists of true values
        rotTrueList.append(rotTrue)
        magTrueList.append(magTrue)

        pupil_mask_fn = os.path.join(
            OUT_PATH,
            'spam',
            'spm_calib_spots_n309_mag%.3f_clk%.3fdeg.fits' % (magTrue, rotTrue)
        )
        pupil_mask_array = fits.getdata(pupil_mask_fn)

        # nActs = 48
        # dm1 = np.zeros((nActs, nActs))
        # dm2 = np.zeros((nActs, nActs))

        print("Computing pupil image")
        params = {
            'pupil_mask_array': pupil_mask_array,
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

        field_cube, _ = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        field = np.mean(field_cube, axis=0)
        field = insertinto(field, imshape)
        amp = np.abs(field)
        image0 = amp**2
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
        # Test that the unmasked pupil parameters are fitted correctly.
        fn_tuning = os.path.join(OUT_PATH, 'spam', 'params_fit_spam_mag_clocking_band%s.yaml' % bandpass.lower())
        # fnYAML = os.path.join(IN_PATH, 'any_band', 'params_fit_spam_mag_clocking.yaml')
        # fnYAML = os.path.join(IN_PATH, 'any_band', 'fit_spam_mag_clocking.yaml')
        magEst, rotEst = pupilfit_gsw.fit_shaped_pupil_mag_clocking(image, fn_tuning)
        print('rotEst = %.3f  magEst = %.4f\n' % (rotEst, magEst))

        tuning_dict = loadyaml(fn_tuning)
        diamList.append(tuning_dict['nBeamNom'])
        # nBeamNom = tuning_dict['nBeamNom']
        # nBeam = nBeamNom*magEst
        # diamList.append(nBeam)
        rotEstList.append(rotEst)
        magEstList.append(magEst)

    data = {
        'band': bandList,
        'beamDiamPix': diamList,
        'rotTrue (degrees)': rotTrueList,
        'rotEst (degrees)': rotEstList,
        'rotError (degrees)':
            np.array(rotTrueList) - np.array(rotEstList),
        'magTrue': magTrueList,
        'magEst': magEstList,
        'magError (%)': (np.array(magTrueList) - np.array(magEstList))*100,
    }

    df = pd.DataFrame(data=data)  # , index=row_labels)

    out_name = os.path.join(OUT_PATH, 'spam', 'calib_spam_results.csv')
    df.to_csv(out_name)


if __name__ == '__main__':
    create_tuning_yaml()
    calibrate_spam()
