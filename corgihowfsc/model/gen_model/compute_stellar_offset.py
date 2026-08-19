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
from cal.psffit.psffit import psffit

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def get_stellar_offset():
    """Determine the unmasked PSF's position."""
    flagPlot = True

    cgi_mode = 'excam_efield'
    imshape = (200, 200)
    polaxis = -10  # compute images for mean X+Y polarization
    # use_errors = True
    # star_spectrum = 'a0v'  # 'k5v'
    # star_vmag = 2.0

    v_dm1 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm2_v.fits' ))

    fn_config = os.path.join(IN_PATH, 'any_band', 'params_psffit.yaml')

    # ccd_fn = os.path.join(IN_PATH, 'ccd_params_for_pupil_imaging.yaml')
    # ccd = loadyaml(ccd_fn)
    # ccd.update({'exptime': 0.8})
    ccd = {}

    bandList = ['1a', '1b', '1c',
                '2a', '2b', '2c',
                '3a', '3b', '3c', '3d', '3e',
                '4a', '4b', '4c'
                ]
    xoffsetList = []
    yoffsetList = []

    for bandpass in bandList:

        if '1' in bandpass:
            cor_type = 'hlc_band1'
            fn_ref_base = 'ref_psf_band_1b.fits'
        elif '2' in bandpass:
            cor_type = 'hlc_band2'
            fn_ref_base = 'ref_psf_band_2c.fits'
        elif '3' in bandpass:
            cor_type = 'hlc_band3'
            fn_ref_base = 'ref_psf_band_3c.fits'
        elif '4' in bandpass:
            cor_type = 'hlc_band4'
            fn_ref_base = 'ref_psf_band_4b.fits'
        else:
            raise ValueError(f'bandpass value {bandpass} not accepted here.')
        
        ref_img = fits.getdata(os.path.join(OUT_PATH, 'focal_plane', 'psf', fn_ref_base))

        print("Computing pupil image")
        params = {
            'ccd':{},
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
            plt.pause(0.1)

            # plt.show()


        # ######################################################
        inputs = {
            'source_image': ref_img,  # reference image
            'target_image': image,  # test image
            'config_file': fn_config,
        }
        shifts, amp_ratio = psffit(**inputs)

        # 5. Return values are shifts
        print(f'*** Band = {bandpass} ***')
        print('x-shift (columns) = %.3f pixels' % (shifts[1]))
        print('y-shift (rows) = %.3f pixels' % (shifts[0]))
        print('')

        xoffsetList.append(shifts[1])
        yoffsetList.append(shifts[0])


    data = {
        'band': bandList,
        'xoffset': xoffsetList,
        'yoffset': yoffsetList,
    }

    df = pd.DataFrame(data=data)  # , index=row_labels)

    out_name = os.path.join(OUT_PATH, 'focal_plane', 'stellar_offset_results.csv')
    df.to_csv(out_name)


if __name__ == '__main__':
    get_stellar_offset()
