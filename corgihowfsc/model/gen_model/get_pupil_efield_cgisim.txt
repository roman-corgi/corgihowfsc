#   Copyright 2020 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
""" 
Get the E-field amplitude and phase at tne input pupil

Example usage:
    python get_pupil_efield_cgisim.py
"""
import os

from astropy.io import fits
import matplotlib.pylab as plt
import numpy as np

# from cal.ampthresh.ampthresh import ampthresh
from cal.util.insertinto import insertinto as inin

import cgisim as cgisim
import proper

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def gen_efield(bandpass, v_dm1, v_dm2):
    cgi_mode = 'excam_efield' #'excam'
    polaxis = -10. # compute images for mean X+Y polarization (don't compute at each polarization)
    outshape = (340, 340)

    if bandpass.lower() == '1b':
        lamc = 0.575        # central wavelength of band
        cor_type = 'hlc'
    elif bandpass.lower() == '2c':
        lamc = 0.6563
        cor_type = 'hlc_band2'
    elif bandpass.lower() == '3c':
        lamc = 0.727
        cor_type = 'hlc_band3'
    elif bandpass.lower() == '4b':
        lamc = 0.825
        cor_type = 'hlc_band4'
    else:
        raise ValueError(f'bandpass value {bandpass} is not allowed here.')
    
    poke_map = np.fromfile(os.path.join(DM1_PATH, 'dm1_target_pokegridopen_2024_07_03.bin'), dtype='>f4').reshape((48, 48))
    

    print(f"\nComputing pupil E-fields for Band {bandpass}.")

    folders = ['poked_dm1', 'poked_none', 'poked_dm2']

    for folder in folders:

        dv_dm1 = 0
        dv_dm2 = 0
        use_pinhole = False
        if folder == 'poked_dm1':
            dv_dm1 = poke_map
        elif folder == 'poked_dm2':
            dv_dm2 = poke_map
        elif folder == 'poked_none':
            use_pinhole = True

        params = {
            # 'cor_type':cor_type,
            'ccd':{},
            'use_pupil_lens':1,
            'use_errors':1,
            # 'polaxis':polaxis, #np.abs(polaxis),
            'use_fpm':0,
            'use_lyot_stop':0,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1+dv_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2+dv_dm2,
            }
        field_cube, _ = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
                star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_pupil_image.fits' )
        field = np.mean(field_cube, axis=0)
        field = inin(field, outshape)
        amp = np.abs(field)
        phase = np.angle(field)
        amp_end2end = amp.copy()
        phase_end2end = phase.copy()
        fits.writeto(os.path.join(HERE, 'out', 'pupil', bandpass.lower(), folder, 'amp_end2end.fits'), amp, overwrite=True)
        fits.writeto(os.path.join(HERE, 'out', 'pupil',  bandpass.lower(), folder, 'phase_end2end.fits'), phase, overwrite=True)

        if use_pinhole:

            params['pinhole_diam_m'] = 9.3e-6  # meters
            field_cube, _ = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
                    star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_pupil_ph_image.fits' )
            field = np.mean(field_cube, axis=0)
            field = inin(field, outshape)
            amp = np.abs(field)
            phase = np.angle(field)
            amp_backend = amp.copy()
            phase_backend = phase.copy()
            fits.writeto(os.path.join(HERE, 'out', 'pupil', bandpass.lower(), folder, 'amp_backend.fits'), amp, overwrite=True)
            fits.writeto(os.path.join(HERE, 'out', 'pupil',  bandpass.lower(), folder, 'phase_backend.fits'), phase, overwrite=True)

            phase_frontend = phase_end2end - phase_backend

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,3), constrained_layout=True )

        # im = ax[0].imshow(phase_end2end, cmap='gray' )
        # ax[0].set_title('End-to-end phase')

        # im = ax[1].imshow(phase_backend, cmap='gray' )
        # ax[1].set_title('Back-end phase')

        # im = ax[2].imshow( phase_frontend, cmap='gray' )
        # ax[2].set_title('Front-end phase')

        # plt.show()

if __name__ == '__main__':

    delta_dm1 = 0

    v_dm1 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'hlc_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'hlc_flat_wfe_dm2_v.fits' ))
    gen_efield('1b', v_dm1, v_dm2)

    v_dm1 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'spc-spec_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'spc-spec_flat_wfe_dm2_v.fits' ))
    gen_efield('2c', v_dm1, v_dm2)
    gen_efield('3c', v_dm1, v_dm2)

    v_dm1 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'spc-wide_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(HERE, 'in', 'flatmaps', 'spc-wide_flat_wfe_dm2_v.fits' ))
    gen_efield('4b', v_dm1, v_dm2)
