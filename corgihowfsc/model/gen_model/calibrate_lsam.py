# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Run LSAM calibration using the SPC SPEC Lyot stop pupil image."""
import os
import pathlib
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import pandas as pd

import cgisim as cgisim
import proper

from cal.pupilfit import pupilfit_gsw
from cal.pupilfit import pupilfit_open as pfo
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
YAML_PATH = os.path.join(IN_PATH, 'any_band')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


# FROZEN
FN_MASK_PARAMS = os.path.join(YAML_PATH, 'params_lyot_stop_mask_def_spec.yaml')
DATA_PATH_SPEC = os.path.join(MASK_PATH, 'SPC_20200617_Spec') 
# WAVELENGTH_LUT = loadyaml(path=os.path.join(MODEL_PATH, 'misc', 'subband_center_wavelengths.yaml'))

FN_PUPILFIT_TUNING = os.path.join(YAML_PATH, 'params_fit_unmasked_pupil.yaml')
FN_OFFSET_PARAMS_DEFAULT = os.path.join(YAML_PATH, 'params_fit_pupil_mask_offsets.yaml')
# FN_LYOT_CALIB_DEFAULT = os.path.join(YAML_PATH, 'params_fit_lsam_mag_clocking.yaml')
DATA_PATH_PUPILFIT_DEFAULT = os.path.join(MASK_PATH, 'pupil')


def create_tuning_yaml():
    #bandList = ['1b', '2c', '3c', '4b']
    fn_tuning_0 = os.path.join(IN_PATH, 'any_band', 'params_fit_lsam_mag_clocking_template.yaml')
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
        dictNew['diamPupil'] = float(diam)

        fn_tuning_new = os.path.join(OUT_PATH, 'lsam', 'params_fit_lsam_mag_clocking_band%s.yaml' % bandpass.lower())
        writeyaml(dictNew, fn_tuning_new)

    return None


def calibrate_lsam():
    """Compute the magnification and clocking for LSAM."""
    flagPlot = True

    cgi_mode = 'excam'
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
    diamList = []
    rotList = []
    magList = []
    magModelList = []


    fn_lyot = os.path.join(MASK_PATH, 'SPC_20200617_Spec', 'LS_SPC_20200617_Spec_1000.fits')
    lyot_stop_array = fits.getdata(fn_lyot)

    for bandpass in bandList:

        print("\n*** Band %s ***" % bandpass)

        if bandpass == '1b':
            cor_type = 'spc-wide_band1'
        elif bandpass == '2c':
            cor_type = 'spc-spec_band2'
        elif bandpass == '3c':
            cor_type = 'spc-spec_band3'
        elif bandpass == '4b':
            cor_type = 'spc-wide_band4'
        else:
            raise ValueError(f'bandpass value {bandpass} not accepted here.')


        print("* Computing unmasked image *")
        params = {
            'lyot_stop_array': lyot_stop_array,
            'use_pupil_mask': 0,
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
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        image0 = insertinto(image0, imshape)
        image_unmasked = image0/np.max(image0)

        fn_fits_out = os.path.join(OUT_PATH, 'lsam', 'image_unmasked_band_%s.fits' % bandpass)
        fits.writeto(fn_fits_out, image_unmasked, overwrite=True)


        if flagPlot:
            plt.figure(1)
            plt.clf()
            plt.title(f'Unmasked: Band {bandpass.upper()}')
            plt.imshow(image_unmasked)
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.pause(0.1)

        print("* Computing masked image *")
        params['use_lyot_stop'] = 1
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        image0 = insertinto(image0, imshape)
        image_masked = image0/np.max(image0)

        fn_fits_out = os.path.join(OUT_PATH, 'lsam', 'image_lyot_masked_band_%s.fits' % bandpass)
        fits.writeto(fn_fits_out, image_masked, overwrite=True)

        if flagPlot:
            plt.figure(2)
            plt.clf()
            plt.title(f'Masked: Band {bandpass.upper()}')
            plt.imshow(image_masked)
            plt.colorbar()
            plt.gca().invert_yaxis()
            fn_fig_out = os.path.join(OUT_PATH, 'lsam', 'fig_lyot_masked_band_%s.png' % bandpass)
            plt.savefig(fn_fig_out, bbox_inches='tight', pad_inches=0.1, dpi=300)

            plt.pause(2)
            # plt.show()


        # ######################################################
        # Fit the unmasked pupil first
        fn_tuning_pupil = os.path.join(IN_PATH, 'any_band', 'params_fit_unmasked_pupil.yaml')
        xOffsetPupil, yOffsetPupil, clockEstPupil, diamEstPupil = pfo.fit_unmasked_pupil(
            image_unmasked, fn_tuning_pupil, data_path=DATA_PATH_PUPIL)

        fn_tuning_lyot = os.path.join(OUT_PATH, 'lsam', 'params_fit_lsam_mag_clocking_band%s.yaml' % bandpass.lower())

        # Fit the Lyot stop next
        inputs = {
            'imageUnmasked': image_unmasked,
            'imageMasked': image_masked,
            'xOffsetPupil': xOffsetPupil,
            'yOffsetPupil': yOffsetPupil,
            'fnMaskParams': FN_MASK_PARAMS,
            'fnOffsetParams': FN_OFFSET_PARAMS_DEFAULT,
            'fnLyotCalib': fn_tuning_lyot,
            'data_path': DATA_PATH_SPEC,
        }
        magEst, rotEst = pupilfit_gsw.fit_lyot_stop_mag_clocking(**inputs)
        # magEst, rotEst = pupilfit_gsw.fit_lyot_stop_mag_clocking(
        #     image_unmasked, image_masked, xOffsetPupil, yOffsetPupil,
        #     fnMaskParams, fnOffsetParams, fnLyotCalib,
        #     data_path=LOCAL_PATH,
        #     )
        print('Lyot Plane: rotEst = %.3f  magEst = %.4f' % (rotEst, magEst))

        # tuning_dict = loadyaml(fn_tuning)
        # diamList.append(tuning_dict['nBeamNom'])
        # # nBeamNom = tuning_dict['nBeamNom']
        # # nBeam = nBeamNom*magEst
        # # diamList.append(nBeam)
        # rotEstList.append(rotEst)
        # magEstList.append(magEst)


        # 4. Compute the magnification factor relative to the 1000x1000 reference file.
        #    This involves dividing out the pupil diameter from the YAML config file,
        #    NOT the one just computed by the script fit_unmasked_pupil.py.
        #
        # If you get a magnification of say 0.98 (relative to the beam diameter),
        # then you have to scale by diamPupil/diamHighResMaskRef,
        # where diamPupil is from:
        # cgi-orbit/model/any_band/params_fit_lsam_mag_clocking.yaml
        # and diamHighResMaskRef is from:
        # cgi-orbit/model/any_band/params_lyot_stop_mask_def_nfov.yaml

        params_lyot = loadyaml(fn_tuning_lyot)
        params_mask = loadyaml(FN_MASK_PARAMS)
        mag_lyot_model = magEst * params_lyot['diamPupil'] / params_mask['diamHighResMaskRef']   
        print('mag relative to 1000: %.4f' % mag_lyot_model)

        diamList.append(params_lyot['diamPupil'])
        rotList.append(rotEst)
        magList.append(rotEst)
        magModelList.append(mag_lyot_model)


    data = {
        'band': bandList,
        'beamDiamPix': diamList,
        'clocking': rotList,
        'mag': magList,
        'magModel': magModelList,
    }
    df = pd.DataFrame(data=data)  # , index=row_labels)
    out_name = os.path.join(OUT_PATH, 'lsam', 'calib_lsam_results.csv')
    df.to_csv(out_name)
    print('Data saved to:\n%s' % out_name)


if __name__ == '__main__':
    create_tuning_yaml()
    calibrate_lsam()
