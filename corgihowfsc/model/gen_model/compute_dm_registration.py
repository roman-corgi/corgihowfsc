"""
Compute the DM registration for one or both DMs.

Example usage:
    python compute_dm_registration.py

"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.io import fits

import proper

from cal.pr.util.unwrap import unwrap_segments
from cal.buildmodel.build_dm import add_dm_registration
from cal.util.loadyaml import loadyaml
from cal.dmreg.dmreg import calc_dmreg_from_poke_grid_for_any_pupil
from cal.dmcoalign.calcoffset import calcoffset
import cal.pupilfit.pupilfit_open as pfo
from cal.util.insertinto import insertinto


# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')

# SIT_PATH = os.path.join(HERE.split('cgi-sit')[0], 'cgi-sit')
# UTIL_PATH = os.path.join(SIT_PATH, 'util')
# MODEL_PATH = os.path.join(SIT_PATH, 'tvac', 'aac')
# YAML_PATH = os.path.join(MODEL_PATH, 'any_band')
# MASK_PATH = os.path.join(MODEL_PATH, 'mask_designs')
# TEST_DATA_PATH = os.path.join(UTIL_PATH, 'testdata')
# DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')

from compute_dm_gain_map import compute_dm_gain_map

# Default Values:
TUPLE_READ_PR = ('wfe', 'bmask', 'amp', 'phwr', 'lam')
# CORO_MODE_DEFAULT = 'ignore' #'dmreg'
FN_DMREG_TUNING_DEFAULT = os.path.join(
    IN_PATH, 'any_band', 'params_dmreg_tuning.yaml')
DEFAULT_PARAMS_PUPIL = os.path.join(
    IN_PATH, 'any_band', 'params_fit_unmasked_pupil.yaml')
DEFAULT_PUPIL_PARAMS_REL_PATH = os.path.join(MASK_PATH, 'pupil')
DEFAULT_ALIGN_COEFFS = os.path.join(
    IN_PATH, 'any_band', 'params_dmcoalign_coeffs_20220203T040217.yaml')
DEFAULT_DMCOALIGN_TRANSFORM = os.path.join(
    IN_PATH, 'any_band', 'params_dmcoalign_calcoffset_transforms.yaml')
# DEFAULT_BACKEND_PRNUM = None
# DEFAULT_DELTAZ = 0.340

# Default values
LABEL_DEFAULT = ''
DATA_PATH_BACKEND_DEFAULT = os.path.join(IN_PATH, 'pr', 'prnum_000279')
SUBTRACT_BACKEND_DEFAULT = False
FN_PR_PROP_PARAMS_DEFAULT = os.path.join(
    IN_PATH, 'pr', 'params_read_and_prop_pr_nfov_band1.yaml')


# def get_fn_dmreg(bandpass):

#     fn = os.path.join(DMREG_PATH, f'howfsc_optical_model_dmreg_only_band_{bandpass}.yaml')

#     return fn


def display_dmreg(map_dict_dm, title=None):
    """
    doc string in cal.dmreg.dmreg.calc_dmreg_from_poke_grid
    map_dict : dict
        Dictionary containing the all the outputs (output maps and cost) from
        dr_util.compute_phase_diff_model() as well as the pupil-to-array and
        DM-to-pupil offsets. The dictionary keys of map_dict
        have the following names:
            ph_diff_model, ph_diff_meas, bmask, cost, xOffsetPupil,
            yOffsetPupil, xOffsetPupilFromDM, yOffsetPupilFromDM
        xOffsetPupil and yOffsetPupil are the offsets of the pupil from the
        array center pixel and are directly output by pupilzernikes().
        xOffsetPupilFromDM and yOffsetPupilFromDM are computed by differencing
        the DM-to-array offset from the pupil-to-array offsets.

    """

    rc('text')#, usetex=True)
    ims = [None] * 3  # empty list to hold image handles
    fig, ax = plt.subplots(nrows=2, ncols=2)
    if title is not None:
        fig.suptitle(title)
    ims[0] = ax[0, 0].imshow(map_dict_dm['ph_diff_meas'])  # cmap='gray'
    ax[0, 0].set_title(r'$\Delta$ Phase Measured')
    ims[1] = ax[0, 1].imshow(map_dict_dm['ph_diff_model'])
    ax[0, 1].set_title(r'$\Delta$ Phase Model')
    ims[2] = ax[1, 0].imshow(map_dict_dm['ph_diff_meas'] - map_dict_dm['ph_diff_model'])
    ax[1, 0].set_title('Measure - Model')    

    for hax in ax[0, :]:
        hax.invert_yaxis()
    ax[1, 0].invert_yaxis()

    # set color map limits
    _ = [im.set_clim((-1.8, 1.8)) for im in ims[:-1]]
    ims[2].set_clim((-1.8, 1.8))

    # fourth axes is a table
    table_offsets = [None, None]*4
    list_str = ['Offset Pup X', 'Offset Pup Y', 'Offset Pup - DM X', 'Offset Pup - DM Y']
    list_val = ['%.3f' % map_dict_dm[a] for a in [
        'xOffsetPupil', 'yOffsetPupil', 'xOffsetPupilFromDM', 'yOffsetPupilFromDM']
        ]
    table_offsets = [[ss, val] for ss, val in zip(list_str, list_val)]
    atable = ax[1, 1].table(
        cellText=table_offsets, loc='center', colWidths=[0.5, 0.25])
    ax[1, 1].axis('off')

    atable.set_fontsize(28)
    atable.scale(1.25, 1.25)

    cbar = plt.colorbar(ims[0], ax=ax[0, :])
    cbar.ax.set_title('Phase (rad)')
    cbar = plt.colorbar(ims[2], ax=ax[1, 0])
    cbar.ax.set_title('Phase (rad)')

    return fig, ax


# def dm_registration_for_open_pupil(
#         prnum_poked_dm1,
#         prnum_unpoked,
#         prnum_poked_dm2,
#         fn_dmreg_tuning=FN_DMREG_TUNING_DEFAULT,
#         fn_pr_prop_params=FN_PR_PROP_PARAMS_DEFAULT,
#         subtract_backend=SUBTRACT_BACKEND_DEFAULT,
#         data_path_backend=DATA_PATH_BACKEND_DEFAULT,
#         update_yaml='ask',
#         # coro_mode=CORO_MODE_DEFAULT,
#         show_plot=False,
#         block=True,
#         ):
#     """Perform DM registration for an open pupil."""
#     # Fit the pupil in the amp map to get back the pupil diameter and offsets.
#     pr_dict_unpoked = get_cleaned_pr_data(
#         prnum_unpoked,
#         fn_pr_prop_params=fn_pr_prop_params,
#         subtract_backend=subtract_backend,
#         data_path_backend=data_path_backend,
#         )
#     amp_unpoked = pr_dict_unpoked['amp']
#     xOffsetPupil, yOffsetPupil, _, diamPupil = pfo.fit_unmasked_pupil(
#         pupil=amp_unpoked**2, fn_tuning=DEFAULT_PARAMS_PUPIL, data_path=DATA_PATH_PUPIL)

#     return dm_registration(
#         prnum_poked_dm1,
#         prnum_unpoked,
#         prnum_poked_dm2,
#         diamPupil,
#         xOffsetPupil,
#         yOffsetPupil,
#         fn_dmreg_tuning=fn_dmreg_tuning,
#         fn_pr_prop_params=fn_pr_prop_params,
#         subtract_backend=subtract_backend,
#         data_path_backend=data_path_backend,
#         update_yaml=update_yaml,
#         # coro_mode=coro_mode,
#         show_plot=show_plot,
#         block=block,
#         )


def dm_registration(
        bandpass,
        fn_dmreg_tuning=FN_DMREG_TUNING_DEFAULT,
        # fn_pr_prop_params=FN_PR_PROP_PARAMS_DEFAULT,
        subtract_backend=SUBTRACT_BACKEND_DEFAULT,
        data_path_backend=DATA_PATH_BACKEND_DEFAULT,
        update_yaml=True, #'ask',
        # coro_mode=CORO_MODE_DEFAULT,
        show_plot=False,
        block=True,
        ):
    """
    Perform DM registration for either an open pupil or a shaped pupil.

    update_yaml can be: 'ask', 'True', or 'False'

    Returns:
      map_dict_dm1, map_dict_dm2: each a dict with kwds 'ppact_cx', 'ppact_cy', 'dx', 'dy', 'thact'
      excam_dx, excam_dy: DM2-to-DM1 LoS Offset (DI+EXCAM pixels)

    """
    if bandpass == '1b':
        lam = 575e-9
    elif bandpass == '2c':
        lam = 660e-9
    elif bandpass == '3c':
        lam = 730e-9
    elif bandpass == '4b':
        lam = 825e-9

    pr_path = os.path.join(GEN_MODEL_PATH, 'out', 'pupil', bandpass, 'poked_none')
    amp = fits.getdata(os.path.join(pr_path, 'amp_end2end.fits'))
    ph_end2end = fits.getdata(os.path.join(pr_path, 'phase_end2end.fits'))
    ph_backend = insertinto(fits.getdata(os.path.join(pr_path, 'phase_backend.fits')), ph_end2end.shape)
    ph_frontend_none = ph_end2end - ph_backend

    pr_path = os.path.join(GEN_MODEL_PATH, 'out', 'pupil', bandpass, 'poked_dm1')
    ph_end2end = fits.getdata(os.path.join(pr_path, 'phase_end2end.fits'))
    ph_frontend_dm1 = ph_end2end - ph_backend

    pr_path = os.path.join(GEN_MODEL_PATH, 'out', 'pupil', bandpass, 'poked_dm2')
    ph_end2end = fits.getdata(os.path.join(pr_path, 'phase_end2end.fits'))
    ph_frontend_dm2 = ph_end2end - ph_backend

    fn_dmreg = os.path.join(GEN_MODEL_PATH, 'out', 'dm', f'howfsc_optical_model_dmreg_only_band_{bandpass}.yaml')
    # fn_dmreg = get_fn_dmreg(bandpass)

    do_dm1_reg = True #False if prnum_poked_dm1 is None else True
    do_dm2_reg = True #False if prnum_poked_dm2 is None else True

    # run pupil fit through dorba
    xOffsetPupil, yOffsetPupil, clockEst, diamPupil = pfo.fit_unmasked_pupil(
        pupil=amp, fn_tuning=DEFAULT_PARAMS_PUPIL, data_path=DATA_PATH_PUPIL)

    # 1. Read the pr amp and phase files.
    amp_unpoked = amp
    ph_unpoked = ph_frontend_none

    fn_dm1_abs = os.path.join(GEN_MODEL_PATH, 'out', 'dm', 'hlc_flat_wfe_dm1_v.fits')
    fn_dm2_abs = os.path.join(GEN_MODEL_PATH, 'out', 'dm', 'hlc_flat_wfe_dm2_v.fits')
    dmv1_nom = proper.prop_fits_read(fn_dm1_abs)
    dmv2_nom = proper.prop_fits_read(fn_dm2_abs)
    nact_dm1, _ = dmv1_nom.shape  # needed in build model below
    nact_dm2, _ = dmv2_nom.shape  # needed in build model below

    # Compute and write new DM gain maps
    _ = compute_dm_gain_map(1, fn_dm1_abs, show_plot=False)
    _ = compute_dm_gain_map(2, fn_dm2_abs, show_plot=False)
        # which_dm,
        # fn_dmabs_map,
        # threshold=THRESHOLD_DEFAULT,
        # out_path=None,
        # out_file=None,
        # show_plot=True,

    poke_map = np.fromfile(os.path.join(GEN_MODEL_PATH, 'out', 'dm', 'dm1_target_pokegridopen_2024_07_03.bin'), dtype='>f4').reshape((48, 48))


    if do_dm1_reg:

        delta_v_dm1_poked = poke_map
        ph_diff_meas_dm1 = ph_frontend_dm1

    if do_dm2_reg:

        delta_v_dm2_poked = poke_map
        ph_diff_meas_dm2 = ph_frontend_dm2
    
    if do_dm1_reg:
        param_dict_dm1, map_dict_dm1 = calc_dmreg_from_poke_grid_for_any_pupil(
            amp_unpoked=amp_unpoked,
            ph_diff_meas=ph_diff_meas_dm1,
            delta_v=delta_v_dm1_poked,
            which_dm=1,
            lam=lam,
            fn_dmreg_def=(fn_dmreg),
            fn_dmreg_tuning_params=(fn_dmreg_tuning),
            diamPupil=diamPupil,
            xOffsetPupil=xOffsetPupil,
            yOffsetPupil=yOffsetPupil,
        )

        # The five parameters are all type numpy.flat64,
        # but writeyaml cannot handle numpy types.
        for kk in param_dict_dm1:
            param_dict_dm1[kk] = float(param_dict_dm1[kk])

        # return param_dict does not include flipx, need to read
        flipx = loadyaml(path=fn_dmreg)['dms']['DM1']['registration']['flipx']
        param_dict_dm1.update({'flipx': flipx})

        if show_plot:
            display_dmreg(map_dict_dm1, title='DM1 Best-Fit Registration')

    if do_dm2_reg:

        param_dict_dm2, map_dict_dm2 = calc_dmreg_from_poke_grid_for_any_pupil(
            amp_unpoked=amp_unpoked,
            ph_diff_meas=ph_diff_meas_dm2,
            delta_v=delta_v_dm2_poked,
            which_dm=2,
            lam=lam,
            fn_dmreg_def=(fn_dmreg),
            fn_dmreg_tuning_params=(fn_dmreg_tuning),
            diamPupil=diamPupil,
            xOffsetPupil=xOffsetPupil,
            yOffsetPupil=yOffsetPupil,
        )

        # The five parameters are all type numpy.flat64,
        # but writeyaml cannot handle numpy types.
        for kk in param_dict_dm2:
            param_dict_dm2[kk] = float(param_dict_dm2[kk])

        # return param_dict does not include flipx, need to read
        flipx = loadyaml(path=fn_dmreg)['dms']['DM2']['registration']['flipx']
        param_dict_dm2.update({'flipx': flipx})

        if show_plot:
            display_dmreg(map_dict_dm2, title='DM2 Best-Fit Registration')

    # 7. Print out the results to the screen:
    #   DM1 registration
    #   DM2 registration
    #   Pupil fit results
    if do_dm1_reg:
        print('\nDM1 Registration:')
        for kk in param_dict_dm1:         
            print(f'{kk:>12}: {param_dict_dm1[kk]:.3f}') if isinstance(param_dict_dm1[kk], float) \
                else print(f'{kk:>12}: {param_dict_dm1[kk]}')

    if do_dm2_reg:
        print('\nDM2 Registration:')
        for kk in param_dict_dm2:
            print(f'{kk:>12}: {param_dict_dm2[kk]:.3f}') if isinstance(param_dict_dm2[kk], float) \
                else print(f'{kk:>12}: {param_dict_dm2[kk]}')

    if do_dm1_reg and do_dm2_reg:
        # print offset between DM1 and DM2 aignment.
        # This is the input to compute_dmcoalign_EXCAM_offset.py
        dm2_dm1_x = param_dict_dm2['dx'] - param_dict_dm1['dx']
        dm2_dm1_y = param_dict_dm2['dy'] - param_dict_dm1['dy']

        print('\nDM2 - DM1 Alignment Offset (PIL+EXCAM pixels):')
        print(f'{"EXCAM x":>12}: {dm2_dm1_x:.3f}')
        print(f'{"EXCAM y":>12}: {dm2_dm1_y:.3f}')

    # 6. update_yaml can be: 'ask', 'True', or 'False'
    #    'ask': get user input to confirm
    #    'False': skip update yaml without asking
    #    'True': update yaml without asking
    if do_dm1_reg:
        update = 'y' if update_yaml == 'True' else 'n'  # initialize variable
        if update_yaml == 'ask':
            update = input("Update DM1 Registration? ('y' or 'n') ")

        if update_yaml == 'True' or update == 'y':
            add_dm_registration(fn=fn_dmreg, dm_id='DM1', nact=nact_dm1, **param_dict_dm1)
            print(f'Updated DM1 registration in {fn_dmreg}')

        else:
            print('Skipping update DM1 registration')

    if do_dm2_reg:
        update = 'y' if update_yaml == 'True' else 'n'  # initialize variable
        if update_yaml == 'ask':
            update = input("Update DM2 Registration? ('y' or 'n') ")

        if update_yaml == 'True' or update == 'y':
            add_dm_registration(fn=fn_dmreg, dm_id='DM2', nact=nact_dm2, **param_dict_dm2)
            print(f'Updated DM2 registration in {fn_dmreg}')

        else:
            print('Skipping update DM2 registration')

    # 9. Call dmcoalign.calcoffset
    if do_dm1_reg and do_dm2_reg:
        excam_dx, excam_dy = calcoffset(
            dm2_dm1_x=dm2_dm1_x,
            dm2_dm1_y=dm2_dm1_y,
            fn_transforms=(DEFAULT_DMCOALIGN_TRANSFORM),
            coeff_yml=(DEFAULT_ALIGN_COEFFS),
        )

        print('\nDM2-to-DM1 LoS Offset (DI+EXCAM pixels):')
        print(f'{"EXCAM x":>12}: {excam_dx:.3f}')
        print(f'{"EXCAM y":>12}: {excam_dy:.3f}')

    else:
        excam_dx = None
        excam_dy = None

    # 8. If requested, display graphs for sanity checking
    if show_plot:
        if block:
            print('Close figure to continue ...')
            plt.show(block=True)
        else:
            plt.show(block=False)

    return map_dict_dm1, map_dict_dm2, excam_dx, excam_dy


#######################################################
if __name__ == '__main__':

    _ = dm_registration('1b', update_yaml='True', show_plot=True)
    _ = dm_registration('2c', update_yaml='True', show_plot=True)
    _ = dm_registration('3c', update_yaml='True', show_plot=True)
    _ = dm_registration('4b', update_yaml='True', show_plot=True)
