"""
Populate the optical model definition file and supporting files for corgihowfsc.

Everything in the 'dms' key will be left untouched for empirical HOWFSC, except
for the gain map file name, which will be updated based on which starting DM
setting is used.

NOTES on data types:
    Refer to https://note.nkmk.me/en/python-numpy-dtype-astype/
    i1 is uint8
    f4 is float32 (f for float, and 4 for 4 bytes)
    < = little-endian (LSB first)
    > = big-endian (MSB first)

Example Calls in a Bash Terminal:
python write_model_for_corgihowfsc.py nfov 1 360deg
python write_model_for_corgihowfsc.py specrot 3 130deg


"""
import argparse
import os
import re
import shutil
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Paths
USER = os.environ.get('USER')

from cal.maskgen import maskgen
from cal.buildmodel.darkhole import gen_dark_hole_from_yaml
from cal.buildmodel import build_dm, build_sls
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml
from cal.dmreg import dr_util as dru
from eetc.tools import get_effective_wavelength
from cal.util import constrain_dm, dmshapes
import cal.pupilfit.pupilfit_open as pfo

# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
DM_SEED_PATH = os.path.join(MODEL_PATH, 'dm_seeds')
IN_PATH = os.path.join(MODEL_PATH, 'gen_model', 'in')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
PARAM_ABS_PATH = os.path.join(IN_PATH, 'any_band')
EVERY_PATH = os.path.join(MODEL_PATH, 'every_mask_config')

# cgi-sit utils
print(GEN_MODEL_PATH)
sys.path.append(GEN_MODEL_PATH)
from compute_dm_gain_map import compute_dm_gain_map
from clean_pr_data import clean_pr_cgisim
from write_cstrat_file_example import write_cstrat_file

# Default Values:
FT_DIR = 'reverse'
MAG_OF_MASK_AT_FPAM = 1.0
MAG_OF_MASK_AT_FSAM = 1.0
DEFAULT_PARAMS_PUPIL = os.path.join(PARAM_ABS_PATH, 'params_fit_unmasked_pupil.yaml')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')


def every_path(fn_setup):

    return os.path.join(MODEL_PATH, 'every_mask_config')


def any_path(fn_setup):

    setup_dict = loadyaml(fn_setup)
    path = os.path.join(MODEL_PATH, setup_dict['coro_mode_parent'], 'any')
    os.makedirs(path, exist_ok=True)

    return path


def specific_path(fn_setup):

    setup_dict = loadyaml(fn_setup)
    path = os.path.join(MODEL_PATH, setup_dict['coro_mode_parent'], setup_dict['coro_mode'])
    os.makedirs(path, exist_ok=True)

    return path


def extract_first_number(text):
    """Extract the first integer found in a string."""
    match = re.search(r'[-+]?\d+', text)
    if match:
        return int(match.group(0))
    
    return None


def get_lam_list(fn_setup):
    AtoM = 1e-10  # angstroms to meters
    setup_dict = loadyaml(fn_setup)
    # if setup_dict['noprobe']:
    #     # If not probing, then cover the whole bandwidth
    #     lam_list = list(setup_dict['lam_central']*np.linspace(1-setup_dict['bw']/2, 1+setup_dict['bw']/2, setup_dict['n_subband']))
    # else:
    #     # If probing, then use the required subbands
    #     lam_list = AtoM * np.array(get_effective_wavelength(cfam=setup_dict['subband_list'], spt=setup_dict['spectral_type']))
    lam_list = AtoM * np.array(get_effective_wavelength(cfam=setup_dict['subband_list'], spt=setup_dict['spectral_type']))

    return lam_list


def add_fixedbp(fn_setup):
    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    data = np.zeros((1024, 1024), dtype='>u1')

    fn = os.path.join(every_path(fn_setup), 'fixedbp_zeros.fits')
    fits.writeto(fn, data, overwrite=True)


def add_pixelweights(fn_setup):
    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    n_subband = setup_dict['n_subband']
    ones_2d = np.ones((setup_dict['n_excam'], setup_dict['n_excam']))

    hdul = fits.HDUList()

    for ii in range(n_subband):
        if ii == 0:
            # Primary HDU, can be empty or contain data
            hdul.append(fits.PrimaryHDU(ones_2d))
        else:
            # Create an ImageHDU for array1 and append it
            image_hdu = fits.ImageHDU(data=ones_2d, name=f'EXTENSION{ii}')
            hdul.append(image_hdu)

    # hdul = fits.PrimaryHDU(ones_2d)
    fn_pw = os.path.join(every_path(fn_setup), f"pixelweights_ones_nlam{n_subband}_nrow{setup_dict['n_excam']}.fits")
    hdul.writeto(fn_pw, overwrite=True)


def setup_steps(fn_setup):
    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    # os.makedirs(any_path(fn_setup), exist_ok=True)
    # os.makedirs(specific_path(fn_setup), exist_ok=True)
        
    # Copy setup YAML file to coro_mode direction
    shutil.copy(fn_setup, specific_path(fn_setup))

    # Copy over the file from the experimental data folder if using the model
    # and if it doesn't exist yet
    print(setup_dict['fn_ref_howfsc_yaml'])
    model_dict = loadyaml(path=(setup_dict['fn_ref_howfsc_yaml']))
    # if setup_dict['fn_ref_howfsc_yaml'] is None:
    #     model_dict = loadyaml(path=(os.path.join(full_path(setup_dict['coro_mode']), setup_dict['fn_howfsc_yaml'])))
    # else:
    #     model_dict = loadyaml(path=(setup_dict['fn_ref_howfsc_yaml']))

    # Just keep the 'dms' dictionary and start from scratch for the rest
    # so that the number of wavelengths is correct and no extra stuff gets kept
    # in 'sls'
    # Don't fail if re-running and the dictionary keys don't exist yet.
    try:
        model_dict.pop('init')
    except:
        pass

    try:
        model_dict.pop('sls')
    except:
        pass

    fn_homf = os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])
    writeyaml(outdict=model_dict, path=fn_homf)

    # Create folders for all the subbands
    for ii in range(setup_dict['n_subband']):

        fn_subband_folder = os.path.join(specific_path(fn_setup), ('subband%d' % ii))
        os.makedirs(fn_subband_folder, exist_ok=True)

        fn_subband_folder = os.path.join(any_path(fn_setup), ('subband%d' % ii))
        os.makedirs(fn_subband_folder, exist_ok=True)

    return None             


def save_gain_and_dm_maps(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    # 3. Get DM1 and DM2 voltage maps for each.
    #    Use the utilities script to pull the DM commands from TDS at the time of nom_prnum.
    dmv1_nom = fits.getdata(setup_dict['fn_dm1v_epup'])
    dmv2_nom = fits.getdata(setup_dict['fn_dm2v_epup'])

    # 3.a. Compute the DM1 and DM2 gain maps for the nominal DM settings. These calls write
    #    gain maps to file, which are then used in the dmreg calculation.
    fn_gain_map_dm1_0 = compute_dm_gain_map(1, dmv1_nom, show_plot=False)
    fn_gain_map_dm2_0 = compute_dm_gain_map(2, dmv2_nom, show_plot=False)

    gain_map_dm1 = fits.getdata(fn_gain_map_dm1_0)
    gain_map_dm2 = fits.getdata(fn_gain_map_dm2_0)

    fn_gain_map_dm1 = os.path.join(any_path(fn_setup), 'gain_map_dm1.fits')
    fn_gain_map_dm2 = os.path.join(any_path(fn_setup), 'gain_map_dm2.fits')

    fits.writeto(fn_gain_map_dm1, gain_map_dm1, overwrite=True)
    fits.writeto(fn_gain_map_dm2, gain_map_dm2, overwrite=True)

    for dm in [1, 2]:
        dm_id = 'DM%d' % dm

        dm_temp = loadyaml(path=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])))

        build_dm.add_dm_arrays(
            fn=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
            dm_id=dm_id,
            gainfn=('../any/gain_map_dm%d.fits' % (dm)),
            flatfn=dm_temp['dms'][dm_id]['voltages']['flatfn'],
            tiefn=dm_temp['dms'][dm_id]['voltages']['tiefn'],
            crosstalkfn=dm_temp['dms'][dm_id]['voltages']['crosstalkfn'],
            )

    dm1v_base = 'dmabs_init_dm1.fits'
    dm2v_base = 'dmabs_init_dm2.fits'
    fits.writeto(os.path.join(any_path(fn_setup), dm1v_base), dmv1_nom, overwrite=True)
    fits.writeto(os.path.join(any_path(fn_setup), dm2v_base), dmv2_nom, overwrite=True)

    build_sls.add_dminit(fn_model=os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml']), dm_id="DM1", dminit='../any/'+dm1v_base)
    build_sls.add_dminit(fn_model=os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml']), dm_id="DM2", dminit='../any/'+dm2v_base)

    return None


def do_epup_and_lyot(fn_setup):
    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    # NXFRESNEL = 2048
    # deltaz_dms = 1.0
    # diam_pupil_pix = 300.2
    # diam_pup_m = 46.3e-3
    # pixpermeter = diam_pupil_pix/diam_pup_m
    # ppl = 2.262
    amp = fits.getdata(setup_dict['fn_amp_end2end'])
    ph_end2end = fits.getdata(setup_dict['fn_ph_end2end'])
    ph_backend = fits.getdata(setup_dict['fn_ph_backend'])

    x_offset_pupil, y_offset_pupil, _, diam_pupil = pfo.fit_unmasked_pupil(
        pupil=amp**2, fn_tuning=DEFAULT_PARAMS_PUPIL, data_path=DATA_PATH_PUPIL)

    # Define the backend aperture as the Lyot stop in order to remove piston/tip/tilt from the backend WFE.
    fn_lyot_full = os.path.join(MASK_PATH, setup_dict['mask_folder'], setup_dict['fn_lyot'])
    lyot = downsample(maskIn=fits.getdata(fn_lyot_full),
                      rotDeg=0, #lyot_dict['clocking_lyot_all'],
                      mag=setup_dict['mag_pupil_vs_ref'], #lyot_dict['lyot_mag_vs_ref'],
                      xOffset=x_offset_pupil,
                      yOffset=y_offset_pupil,
                      padFac=1.2,
                      flipx=False,
                      )
    lyot = inin(lyot, ph_end2end.shape)
    ls_thresh = 1e-3
    pupil_backend = np.zeros(ph_end2end.shape)
    pupil_backend[lyot > ls_thresh] = 1

    amp, ph_frontend, ph_backend, bmask_pupil, lam = clean_pr_cgisim(
        amp, ph_end2end, ph_backend, setup_dict['cal_bandpass'], pupil_backend=pupil_backend)

    N0 = amp.shape[0]

    # Pupil masks
    # gen_zero_or_one_pupil_masks(fn_setup, amp.shape)  # Only need to do once
    wfe_from_spm = np.zeros_like(ph_frontend)
    add_spm(fn_setup, diam_pupil, x_offset_pupil, y_offset_pupil, ph_frontend.shape, wfe_from_spm, setup_dict['lam_central'])
    gen_lyot_stop(fn_setup, ph_backend, diam_pupil, x_offset_pupil, y_offset_pupil, setup_dict['lam_central'])

    # Write pupil amplitude FITS file
    # Assume achromatic pupil amplitude, so don't put in chromatic subfolders
    base_amp_fn = 'epup_amp.fits'
    fn_amp = os.path.join(any_path(fn_setup), 'epup_amp.fits')
    hdu = fits.PrimaryHDU(amp)
    hdu.writeto(fn_amp, overwrite=True)

    for index, lam in enumerate(lam_list):
        print('Doing epup for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

        # Have WFE scale inversely with wavelength
        wfe_temp = ph_frontend * (setup_dict['lam_central'] / lam)
        fn_wfe = os.path.join(any_path(fn_setup), ('subband%d' % index), 'epup_ph.fits')
        fits.writeto(fn_wfe, wfe_temp, overwrite=True)

        build_sls.add_epup(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                           lam_index=index,
                           afn=base_amp_fn,
                           pdp=float(diam_pupil),
                           pfn = '../any/' + os.path.join(('subband%d' % index), 'epup_ph.fits'),
                           tip=0,
                           tilt=0,
                           )
        print('done.')

    return x_offset_pupil, y_offset_pupil, _, diam_pupil


def gen_lyot_stop(fn_setup, wfe_backend, diam_pupil, x_offset_pupil, y_offset_pupil, wavelength_pr):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    # fn_lyot_meas = os.path.join(PARAM_ABS_PATH, 'params_meas_lsam_calib.yaml')
    # lyot_dict = loadyaml(path=(fn_lyot_meas))

    fn_lyot_full = os.path.join(MASK_PATH, setup_dict['mask_folder'], setup_dict['fn_lyot'])
    lyot = downsample(maskIn=fits.getdata(fn_lyot_full),
                      rotDeg=0, #lyot_dict['clocking_lyot_all'],
                      mag=setup_dict['mag_pupil_vs_ref'], #lyot_dict['lyot_mag_vs_ref'],
                      xOffset=x_offset_pupil,
                      yOffset=y_offset_pupil,
                      padFac=1.2,
                      flipx=False,
                      )

    # Lyot stop amplitude is wavelength invariant, so put one folder up.
    lyot = inin(arr0=lyot, outsize=wfe_backend.shape)
    fn_out = os.path.join(any_path(fn_setup), 'lyot_amp.fits')
    fits.writeto(fn_out, lyot, overwrite=True)
    
    # # Remove piston, tip, and tilt within the opening of the Lyot stop from the
    # # backend phase map.
    # wfe_backend_no_ptt = remove_ptt_by_psf_centering(lyot, wfe_backend) * lyot
    # wfe_backend_no_ptt = dru.remove_piston_tip_tilt(arrayToFit=wfe_backend, mask=np.round(lyot).astype(bool)) 
    wfe_backend_no_ptt = wfe_backend

    # Need a different backend phase at each wavelength
    for index, lam in enumerate(lam_list):
        print('Adding Lyot stop for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

        wfe_backend_temp = wfe_backend_no_ptt * (wavelength_pr / lam)

        fn_wfe = os.path.join(any_path(fn_setup), ('subband%d' % index), 'lyot_ph.fits')
        fits.writeto(fn_wfe, wfe_backend_temp, overwrite=True)

        build_sls.add_lyot_mask(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                                lam_index=index,
                                afn='../any/lyot_amp.fits',
                                pdp=float(diam_pupil),
                                pfn='../any/' + os.path.join(('subband%d' % index), 'lyot_ph.fits'),
                                )
        print('done.')

        pass

    return None


def gen_field_stop(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    for index, lam in enumerate(lam_list):
        print('Adding field stop for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

        # if setup_dict['noprobe']:
        #     tip = 0
        #     tilt = 0
        # else:
        #     tip = float(setup_dict['tip_list'][index])
        #     tilt = float(setup_dict['tilt_list'][index])

        tip = float(setup_dict['tip_list'][index])
        tilt = float(setup_dict['tilt_list'][index])

        if setup_dict['fn_fs'] is None:

            build_sls.add_fs(fn_model = (os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                             lam_index = index,
                             afn = '../../every_mask_config/ones_like_fs.fits',
                             pfn = '../../every_mask_config/zeros_like_fs.fits',
                             ppl=float(setup_dict['ppl_meas_ref']*lam/setup_dict['lam_central']),
                             )

        else:

            fs_amp_base_fn = os.path.join(('subband%d' % index), 'fs_amp.fits')
            build_sls.add_fs(fn_model = (os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                             lam_index = index,
                             afn = fs_amp_base_fn,
                             pfn = '../../every_mask_config/zeros_like_fs.fits',
                             ppl=float(setup_dict['ppl_meas_ref']*lam/setup_dict['lam_central']),
                             )

            fn_fs_full = os.path.join(MASK_PATH, setup_dict['mask_folder'], setup_dict['fn_fs'])
            mask = downsample(maskIn=fits.getdata(fn_fs_full),
                              rotDeg=setup_dict['fs_rot_deg'],
                              mag=MAG_OF_MASK_AT_FSAM*setup_dict['ppl_meas_ref']/setup_dict['ppl_fs_ref'],
                              xOffset=tip+setup_dict['fs_x_offset'],
                              yOffset=tilt+setup_dict['fs_y_offset'],
                              padFac=1.2,
                              flipx=False,
                              )

            mask = inin(arr0=mask, outsize=(setup_dict['n_excam'], setup_dict['n_excam']))

            fn_out = os.path.join(specific_path(fn_setup), fs_amp_base_fn)
            fits.writeto(fn_out, np.real(mask), overwrite=True)

            print('done.')

    return None


def do_dark_hole_and_star_position(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    for index, lam in enumerate(lam_list):

        # if setup_dict['noprobe']:
        #     tip = 0
        #     tilt = 0
        # else:
        #     tip = float(setup_dict['tip_list'][index])
        #     tilt = float(setup_dict['tilt_list'][index])
        tip = float(setup_dict['tip_list'][index])
        tilt = float(setup_dict['tilt_list'][index])

        print('Adding downstream tip/tilt for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

        ## Lyot tip/tilt
        build_sls.add_lyot_tip_tilt(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                                    lam_index=index,
                                    tip=tip,
                                    tilt=tilt,
                                    )

        # Dark Hole Software Mask
        if setup_dict['dh_match_fs']:

            # Use a thresholded version of the field stop
            fs_amp_base_fn = os.path.join(('subband%d' % index), 'fs_amp.fits')
            fn_fs_full = os.path.join(specific_path(fn_setup), fs_amp_base_fn)
            dh = fits.getdata(fn_fs_full)
            dh_mask = np.zeros_like(dh)
            dh_mask[dh > setup_dict['fs_thresh_for_dh']] = 1


        else: 
            dh_dict_all = {'nRows': setup_dict['n_excam'],
                    'nCols': setup_dict['n_excam'],
                    'shapes': {'shape0': setup_dict['dh_dict']}
                    }
            dh_dict_all['shapes']['shape0']['radiusInner'] = setup_dict['ppl_meas_ref']*dh_dict_all['shapes']['shape0']['radiusInnerLamD']
            dh_dict_all['shapes']['shape0']['radiusOuter'] = setup_dict['ppl_meas_ref']*dh_dict_all['shapes']['shape0']['radiusOuterLamD'] 
            dh_dict_all['shapes']['shape0']['xOffset'] = tip + dh_dict_all['shapes']['shape0']['xOffset']
            dh_dict_all['shapes']['shape0']['yOffset'] = tip + dh_dict_all['shapes']['shape0']['yOffset']

            fn_dh_yaml_local = os.path.join(('subband%d' % index), 'dh_sw_mask.yaml')
            fn_dh_yaml = os.path.join(specific_path(fn_setup), fn_dh_yaml_local)
            writeyaml(outdict=dh_dict_all, path=(fn_dh_yaml))
            dh_mask = gen_dark_hole_from_yaml(fnSpecs=(fn_dh_yaml))

        fn_dh_fits_local = os.path.join(('subband%d' % index), 'dh_sw_mask.fits')
        fn_dh_fits = os.path.join(specific_path(fn_setup), fn_dh_fits_local)
        dh_mask = dh_mask.astype('i1')  # convert to uint8 for FITS
        fits.writeto(fn_dh_fits, dh_mask, overwrite=True)

        build_sls.add_dh(fn_model = (os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                         lam_index = index,
                         dh = fn_dh_fits_local,
                         )

        print('done.')

    return None


def add_fpm(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    shapeOut = (setup_dict['n_fpm'], setup_dict['n_fpm'])

    for index, lam in enumerate(lam_list):
        print('Adding fpm for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

        if setup_dict['is_hlc']:

            fnCalibData = os.path.join(IN_PATH, 'any_band', 'params_hlc_occulter_calib_data_cgisim.yaml')
            fnOccData = os.path.join(MASK_PATH, setup_dict['mask_folder'], 'mask_file_data_HLC_Band%d.yaml' % extract_first_number(setup_dict['coro_mode']))
            inputs = {
                'lam': lam,
                'scaleWithWavelength': False,
                'shapeOut': shapeOut,
                'fnCalibData': (fnCalibData),
                'fnOccData': (fnOccData),
                'xOffset': 0,
                'yOffset': 0,
                'use_fourier': False,
                'data_path': os.path.join(MASK_PATH, setup_dict['mask_folder']),
            }
            fpm = maskgen.gen_hlc_occulter(**inputs)
            fpm_phase = np.angle(fpm)

        else:

            if 'specrot' in setup_dict['coro_mode']:
                rotDeg = 0
            else:
                rotDeg = 270

            fn_fpm_full = os.path.join(MASK_PATH, setup_dict['mask_folder'], setup_dict['fn_fpm'])
            fpm = downsample(maskIn=fits.getdata(fn_fpm_full),
                            rotDeg=rotDeg,
                            mag=MAG_OF_MASK_AT_FPAM*setup_dict['ppl_fpm_central']/setup_dict['ppl_fpm_ref'],
                            xOffset=0,
                            yOffset=0,
                            padFac=1.2,
                            flipx=False,
                            )
            fpm = inin(fpm, shapeOut)
            fpm_phase = np.zeros(shapeOut)

        fn_real_out = os.path.join(any_path(fn_setup), ('subband%d' % index), 'fpm_amp.fits')
        hdu = fits.PrimaryHDU(np.abs(fpm))
        hdu.writeto(fn_real_out, overwrite=True)

        fn_imag_out = os.path.join(any_path(fn_setup), ('subband%d' % index), 'fpm_ph.fits')
        hdu = fits.PrimaryHDU(fpm_phase)
        hdu.writeto(fn_imag_out, overwrite=True)

        build_sls.add_fpm(fn_model = (os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                        lam_index = index,
                        afn=os.path.join(('../any/subband%d' % index), 'fpm_amp.fits'),
                        isopen=setup_dict['is_fpm_open'],
                        pfn=os.path.join(('../any/subband%d' % index), 'fpm_ph.fits'),
                        ppl=float(setup_dict['ppl_fpm_central']/MAG_OF_MASK_AT_FPAM*(lam/setup_dict['lam_central'])),
                        )

        print('done.')

    return fpm


# %% Simpler functions and DORBA wrappers

def add_lam(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    for index, lam in enumerate(lam_list):

        build_sls.add_lam(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                          lam_index=index,
                          lam=float(lam),
                          )

    return None


def add_ft_dir(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)

    for index, lam in enumerate(lam_list):

        build_sls.add_ft_dir(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                          lam_index=index,
                          ft_dir=FT_DIR,
                          )

    return None

def add_spm(fn_setup, diam_pupil, x_offset_pupil, y_offset_pupil, out_shape, spm_wfe_central, wavelength_pr):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)
    lam_list = get_lam_list(fn_setup)
    
    if not setup_dict['use_spm']:
        do_no_spm(fn_setup, diam_pupil)
    
    else:

        # fn_spam_meas = os.path.join(PARAM_ABS_PATH, 'params_meas_spam_calib.yaml')
        # spam_dict = loadyaml(path=(fn_spam_meas))   

        spm_amp_base_fn = 'spm_amp.fits'
        fn_spm_full = os.path.join(MASK_PATH, setup_dict['mask_folder'], setup_dict['fn_spm'])
        mask = downsample(maskIn=fits.getdata(fn_spm_full),
                        rotDeg=0, #spam_dict['clocking_spam'],
                        mag=setup_dict['mag_pupil_vs_ref'], #spam_dict['spm_mag_vs_ref'],
                        xOffset=x_offset_pupil,
                        yOffset=y_offset_pupil,
                        padFac=1.2,
                        flipx=False,
                        )
        mask = inin(arr0=mask, outsize=out_shape)
        
        fn_out = os.path.join(any_path(fn_setup), spm_amp_base_fn)
        fits.writeto(fn_out, np.real(mask), overwrite=True)
            
        for index, lam in enumerate(lam_list):
            print('Adding SPM for wavelength %d of %d...' % (index+1, setup_dict['n_subband']), end='')

            spm_wfe_temp = spm_wfe_central * (wavelength_pr / lam)
            fn_wfe = os.path.join(any_path(fn_setup), ('subband%d' % index), 'spm_ph.fits')
            fits.writeto(fn_wfe, spm_wfe_temp, overwrite=True)

            build_sls.add_sp(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                            lam_index=index,
                            afn='../any/'+spm_amp_base_fn,
                            pfn='../any/'+os.path.join(('subband%d' % index), 'spm_ph.fits'),
                            pdp=float(diam_pupil),
                            )

            print('done.')

        return None


def do_no_spm(fn_setup, diam_pupil):
    
    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    for index in range(setup_dict['n_subband']):

        build_sls.add_sp(fn_model=(os.path.join(specific_path(fn_setup), setup_dict['fn_howfsc_yaml'])),
                         lam_index=index,
                         afn='../../every_mask_config/ones_like_pupil.fits',
                         pdp=float(diam_pupil),
                         pfn='../../every_mask_config/zeros_like_pupil.fits',
                         )


def gen_zero_or_one_pupil_masks(fn_setup, pr_shape):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    fn_out_zeros = os.path.join(every_path(fn_setup), 'zeros_like_pupil.fits')
    fn_out_ones = os.path.join(every_path(fn_setup), 'ones_like_pupil.fits')

    hdu = fits.PrimaryHDU(np.zeros((pr_shape)))
    hdu.writeto(fn_out_zeros, overwrite=True)

    hdu = fits.PrimaryHDU(np.ones((pr_shape)))
    hdu.writeto(fn_out_ones, overwrite=True)


def gen_zero_or_one_excam_sized_masks(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    fn_out_zeros = os.path.join(every_path(fn_setup), 'zeros_like_fs.fits')
    fn_out_ones = os.path.join(every_path(fn_setup), 'ones_like_fs.fits')

    hdu = fits.PrimaryHDU(np.zeros((setup_dict['n_excam'], setup_dict['n_excam'])))
    hdu.writeto(fn_out_zeros, overwrite=True)

    hdu = fits.PrimaryHDU(np.ones((setup_dict['n_excam'], setup_dict['n_excam'])))
    hdu.writeto(fn_out_ones, overwrite=True)


def write_cstrat(fn_setup):

    # Load configuration values
    setup_dict = loadyaml(fn_setup)

    coro_mode = setup_dict['coro_mode']
    fn_cstrat = os.path.join(specific_path(fn_setup), f'cstrat_example_{coro_mode}.yaml')

    nlam = setup_dict['n_subband']
    nexcam = setup_dict['n_excam']

    # NOTE: cstrat needs to be tuned by the user
    reg_vec = [-2, -2, -2, -5] + [-2, -3, -3, -6]*5 + [-2, -3, -5, -3, -3, -2] 
    reg_vec = np.asarray(reg_vec)
    dmmultgain_vec = np.ones_like(reg_vec)
    dmmultgain_vec[reg_vec == -6] = 0.75
    write_cstrat_file(fn_cstrat, nlam, nexcam, reg_vec, dmmultgain_vec)


def write_model(model_type, band_num, label):

    fn_setup = os.path.join(MODEL_PATH, 'homf_params', f'homf_params_{model_type}_band{str(band_num)}_{label}.yaml')

    setup_steps(fn_setup)
    # add_fixedbp(fn_setup)  # Only need to do once
    # add_pixelweights(fn_setup)  # Only need to do twice, once for 3 subbands and once for 5 subbands
    save_gain_and_dm_maps(fn_setup)
    add_lam(fn_setup)
    add_ft_dir(fn_setup)
    do_epup_and_lyot(fn_setup)  # Must be after save_gain_and_dm_maps
    # gen_zero_or_one_excam_sized_masks(fn_setup)  # Only need to do once
    gen_field_stop(fn_setup)

    add_fpm(fn_setup)
    do_dark_hole_and_star_position(fn_setup)
    write_cstrat(fn_setup)

#####################################################################
if __name__ == '__main__':

    mode_type_list = ['nfov', 'wfov', 'spec', 'specrot', 'mswc']
    band_list = [1, 2, 3, 4]

    parser = argparse.ArgumentParser(
        prog="python write_model.py",
        description="Populate the optical model definition file and generate the FITS files for it.")
    parser.add_argument("model_type", type=str, choices=mode_type_list, help=f"Which type of mask config to use. Choices are {mode_type_list}.")
    parser.add_argument("band_num", type=int, choices=band_list, help=f"Which bandpass to use. Choices are {band_list}.")
    parser.add_argument("label", type=str,
                        help="Text descriptor used to differentiate the use case from others in the same band and mask config. Typically describes the dark hole size/shape.")

    args = parser.parse_args()

    retvals = write_model(
        args.model_type,
        args.band_num,
        args.label,
    )
