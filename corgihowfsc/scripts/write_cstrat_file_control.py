# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Example script to write cstrat files.

This script focuses on the regularization strategy and 
has hard-coded values for the probing and pixel weights.
"""
import os

import numpy as np

import howfsc
from howfsc.util import check
from howfsc.util.writeyaml import writeyaml


def write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec):

    # Input checks
    reg_vec = check.oneD_array(reg_vec, 'reg_vec', TypeError)
    dmmultgain_vec = check.oneD_array(dmmultgain_vec, 'dmmultgain_vec', TypeError)
    if len(reg_vec) != len(dmmultgain_vec):
        raise ValueError('reg_vec and dmmultgain_vec must have the same length.')
    niter = len(reg_vec)

    # Default probing values for now.
    unprobedsnr0 = [{
        'first': 1,
        'last': None,
        'low': 0,
        'high': None,
        'value': 5,
    }, ]

    probedsnr0 = [{
        'first': 1,
        'last': None,
        'low': 0,
        'high': None,
        'value': 7,
    },]

    probeheight = [
        {
        'first': 1,
        'last': None,
        'low': 0,
        'high': 1.0e-7,
        'value': 1.0e-7,
        },

        {
        'first': 1,
        'last': None,
        'low': 1.0e-7,
        'high': 1.0e-6,
        'value': 1.0e-6,
        },

        {
        'first': 1,
        'last': None,
        'low': 1.0e-6,
        'high': None,
        'value': 1.0e-5,
        },

    ]


    # Real values
    pixelweights = [{
        'first': 1,
        'last': None,
        'low': 0,
        'high': None,
        'value': ('../../every_mask_config/pixelweights_ones_nlam%d_nrow%d.fits' % (nlam, nexcam)),
    },]

    regularization = []
    for iter, reg in enumerate(reg_vec):
        if iter+1 == niter:
            last = None
        else:
            last = iter+1
    
        next_reg = {
        'first': iter+1,
        'last': last,
        'low': 0,
        'high': None,
        'value': float(reg),
        }
        regularization.append(next_reg)

    dmmultgain = []
    for iter, dmg in enumerate(dmmultgain_vec):

        if iter+1 == niter:
            last = None
        else:
            last = iter+1

        next_dmg = {
        'first': iter+1,
        'last': last,
        'low': 0,
        'high': None,
        'value': float(dmg),
        }
        dmmultgain.append(next_dmg)


    # Combine into one dictionary
    all_dict = {}
    all_dict['unprobedsnr'] = unprobedsnr0
    all_dict['probedsnr'] = probedsnr0
    all_dict['probeheight'] = probeheight
    all_dict['fixedbp'] = '../../every_mask_config/fixedbp_zeros.fits'
    all_dict['pixelweights'] = pixelweights
    all_dict['regularization'] = regularization
    all_dict['dmmultgain'] = dmmultgain

    # Write to file
    writeyaml(all_dict, fn)
    print(f'New cstrat file written to:\t\t{fn}')

    return all_dict


if __name__ == "__main__":

    nexcam = 153

    fn = '../model/nfov_band1/nfov_band1_360deg/cstrat_nfov_band1_360deg_alpha01.yaml'
    nlam = 3
    reg_vec = [-2, -2, -2, -2, -5]*6
    reg_vec = np.asarray(reg_vec)
    dmmultgain_vec = np.ones_like(reg_vec)
    dmmultgain_vec[reg_vec == -6] = 0.5
    write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    fn = '../model/nfov_band1/nfov_band1_360deg/cstrat_nfov_band1_360deg_alpha02.yaml'
    nlam = 3
    reg_vec = [-2, -2, -2, -2, -5] + [-2, -2, -3, -2, -5.5] + [-2, -2, -3, -2, -6] + [-2, -2, -4, -2, -6] + [-2, -2, -4, -2, -6] + [-2, -3, -3, -2, -2]
    reg_vec = np.asarray(reg_vec)
    dmmultgain_vec = np.ones_like(reg_vec)
    dmmultgain_vec[reg_vec == -6] = 0.5
    write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    import matplotlib.pyplot as plt

    fn_png = '/Users/ajriggs/Downloads/fig_log10reg_schedule.png'

    plt.figure(1)
    plt.plot(reg_vec, '-bx')
    plt.xlabel('HOWFSC Iteration')
    plt.ylabel('log10(Regularization)')
    plt.savefig(fn_png, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.show()

    fn = '../model/wfov_band4/wfov_band4_360deg/cstrat_wfov_band4_360deg_alpha03.yaml'
    nlam = 3
    reg_vec = [-2, -2, -2, -2, -5] + [-2, -2, -3, -2, -5.5] + [-2, -2, -3, -2, -6] + [-2, -2, -4, -2, -6] + [-2, -2, -4, -2, -6] + [-2, -3, -3, -2, -2]
    reg_vec = np.asarray(reg_vec)
    dmmultgain_vec = np.ones_like(reg_vec)
    # dmmultgain_vec[reg_vec == -6] = 0.5
    write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    # %% Files for getting DM seeds with cgi-howfsc
    # fn = 'output/cstrat_nfov_band1_flat_noprobe.yaml'
    # nlam = 7
    # nexcam = 79
    # reg_vec = [-2, -2, -2, -5] + [-2.5, -2.5, -2.5, -5] + [-3, -3, -3, -5.5] + [-3, -3, -3, -6]*5 + [-3, -3.5, -3.5, -6.5]*3 + [-3, -3.5, -3.5, -7]*8 + [-4, -3, -3, -2] 
    # reg_vec = np.asarray(reg_vec)
    # dmmultgain_vec = np.ones_like(reg_vec)
    # dmmultgain_vec[reg_vec == -6.5] = 0.75
    # dmmultgain_vec[reg_vec == -7] = 0.5
    # write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    # fn = 'output/cstrat_nfov_band1_flat_half_noprobe.yaml'
    # nlam = 7
    # nexcam = 79
    # reg_vec = [-2, -2, -2, -5] + [-2.5, -2.5, -2.5, -5] + [-3, -3, -3, -5.5] + [-3, -3, -3, -6]*5 + [-3, -3.5, -3.5, -6.5]*3 + [-3, -3.5, -3.5, -7]*3 + [-4, -3, -3, -2] 
    # reg_vec = np.asarray(reg_vec)
    # dmmultgain_vec = np.ones_like(reg_vec)
    # dmmultgain_vec[reg_vec == -6.5] = 0.75
    # dmmultgain_vec[reg_vec == -7] = 0.5
    # write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    # fn = 'output/cstrat_wfov_band4_flat_noprobe.yaml'
    # nlam = 5
    # nexcam = 125
    # write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    # fn = 'output/cstrat_spec_band3_flat_noprobe.yaml'
    # nlam = 9
    # nexcam = 79
    # write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    # fn = 'output/cstrat_spec_band2_flat_noprobe.yaml'
    # nlam = 9
    # nexcam = 79
    # write_cstrat_file(fn, nlam, nexcam, reg_vec, dmmultgain_vec)

    pass
