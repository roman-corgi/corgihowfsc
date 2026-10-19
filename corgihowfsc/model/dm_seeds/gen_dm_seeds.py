# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#pylint: disable=line-too-long
"""
Use efc_computation to get model-based DM seeds for each Roman CGI mode using the compact model.

Example usage:
python gen_dm_seeds.py nfov_band1_360deg --niter 80
python gen_dm_seeds.py wfov_band1_360deg --niter 60
python gen_dm_seeds.py wfov_band4_360deg --niter 60
python gen_dm_seeds.py spec_band3_130deg --niter 60
python gen_dm_seeds.py spec_band2_130deg --niter 60
python gen_dm_seeds.py specrot_band3_130deg --niter 60
python gen_dm_seeds.py specrot_band2_130deg --niter 60

"""

import time
import os
import argparse
import logging

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt

import howfsc
from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjtwj import JTWJMap

from howfsc.model.mode import CoronagraphMode

from howfsc.util.loadyaml import loadyaml

from howfsc.gitl import efc_computation
from howfsc.precomp import howfsc_precomputation

HERE = os.path.dirname(os.path.abspath(__file__))
# howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
# OUT_PATH_DEFAULT = os.path.join(HERE, 'output')
# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
DM_SEED_PATH = os.path.join(MODEL_PATH, 'dm_seeds')
IN_PATH = os.path.join(MODEL_PATH, 'gen_model', 'in')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
PARAM_ABS_PATH = os.path.join(IN_PATH, 'any_band')
EVERY_PATH = os.path.join(MODEL_PATH, 'every_mask_config')



if __name__ == "__main__":
    defjacpath = os.path.join(DM_SEED_PATH, 'jac')

    # setup for cmd line args
    ap = argparse.ArgumentParser(prog='python gen_dm_seeds.py',
                                 description="Use efc_computation to get model-based DM seeds for each Roman CGI mode using the compact model.")
    ap.add_argument('mode', type=str, help="Coronagraph mode.")
    ap.add_argument('-n', '--niter', type=int, default=5, help="Number of iterations to run.  Defaults to 5.")
    # ap.add_argument('--logfile', default=None, help="If present, absolute path to file location to log to.")
    ap.add_argument('--precomp', default='precomp_jacs_always', choices=['load_all', 'precomp_all_once', 'precomp_jacs_once', 'precomp_jacs_always'], help="Specifies Jacobian precomputation behavior.  'load_all' will load everything from files at the start and leave them fixed.  'precomp_all_once' will compute a Jacobian, JTWJs, and n2clist once at start.  'precomp_jacs_once' will compute a Jacobian and JTWJs once at start, and load n2clist from file.  'precomp_jacs_always' will compute a Jacobian and JTWJs at start and at every iteration; n2clist will be loaded from file.")
    ap.add_argument("--no_plot", action="store_true", help="Don't display plots of DM voltages and dark hole normalized intensity.")
    ap.add_argument(
        '--num_process', default=0, #None,
        help="number of processes, defaults to None, indicating no multiprocessing. If 0, then howfsc_precomputation() defaults to the available number of cores", type=int
    )
    ap.add_argument(
        '--num_threads', default=None,
        help="set mkl_num_threads to use for parallel processes for calcjacs(). If None (default), then do nothing (number of threads might also be set externally through environment variable 'MKL_NUM_THREADS' or 'HOWFS_CALCJAC_NUM_THREADS'",
        type=int
    )
    # ap.add_argument('--fileout', default=None, help="If present, absolute path to file location of output .fits (including \'.fits\' at the end) file containing framelist and prev_exptime_list, as well as nlam stored as a header.")
    ap.add_argument('-j', '--jacpath', type=str, default=defjacpath, help="absolute path to read Jacobian files from")
    # ap.add_argument('--out_path', default=out_path, help='Directory where output files should be written.')
    args = ap.parse_args()

    otherlist = []
    abs_dm1list = []
    abs_dm2list = []
    prev_c_list = []

    # User params
    niter = args.niter
    mode = args.mode
    # out_path = args.out_path
    # logfile = args.logfile
    # fileout = args.fileout
    jacpath = args.jacpath
    show_plot = not args.no_plot

    modelpath = os.path.join(DM_SEED_PATH, mode)
    out_path = os.path.join(modelpath, 'out')
    os.makedirs(out_path, exist_ok=True)

    logfile = os.path.join(out_path, 'log.txt')
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # # Set up logging
    # if logfile is not None:
    #     logging.basicConfig(filename=logfile, level=logging.INFO)
    #     pass
    # else:
    #     logging.basicConfig(level=logging.INFO)
    #     pass
    log = logging.getLogger(__name__)

    # default croplist values 
    nclean = 1024
    nrow = 153
    ncol = 153
    # lrow = 436
    # lcol = 436

    cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    jacfile = os.path.join(jacpath, 'jac_%s.fits' % mode)
    cstratfile = os.path.join(modelpath, 'cstrat_noprobe_%s.yaml' % mode)

    paramsetupfile = os.path.join(DM_SEED_PATH, 'homf_params', 'homf_params_%s.yaml' % mode)
    params = loadyaml(paramsetupfile)
    nrow = params['n_excam']
    ncol = nrow


    # if mode == 'nfov_band1_360deg':
    #     modelpath = os.path.join(DM_SEED_PATH, mode)
    #     cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    #     jacfile = os.path.join(jacpath, 'jac_%s.fits' % mode)
    #     cstratfile = os.path.join(modelpath, 'cstrat_nfov_band1_360deg.yaml')

    #     paramsetupfile = os.path.join(DM_SEED_PATH, 'homf_params', 'homf_params_%s.yaml' % mode)
    #     params = loadyaml(paramsetupfile)
    #     nrow = params['n_excam']
    #     ncol = nrow
        
    # elif mode == 'nfov_band1_180deg':
    #     modelpath = os.path.join(howfscpath, 'model', 'testdata', 'nfov_band1_180deg')
    #     cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    #     jacfile = os.path.join(jacpath, 'nfov_flat_half_jac.fits')
    #     cstratfile = os.path.join(modelpath, 'cstrat_nfov_band1_180deg.yaml')

    #     paramsetupfile = os.path.join(modelpath, 'homf_params_nfov_band1_180deg.yaml')
    #     params = loadyaml(paramsetupfile)
    #     nrow = params['n_excam']
    #     ncol = nrow
    
    # elif mode == 'spec_band2_130deg':
    #     modelpath = os.path.join(howfscpath, 'model', 'testdata', 'spec_band2_130deg')
    #     cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    #     jacfile = os.path.join(jacpath, 'nfov_jac.fits')
    #     cstratfile = os.path.join(modelpath, 'cstrat_spec_band2_130deg.yaml')

    #     paramsetupfile = os.path.join(modelpath, 'homf_params_spec_band2_130deg.yaml')
    #     params = loadyaml(paramsetupfile)
    #     nrow = params['n_excam']
    #     ncol = nrow

    # elif mode == 'spec_band3_130deg':
    #     modelpath = os.path.join(howfscpath, 'model', 'testdata', 'spec_band3_130deg')
    #     cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    #     jacfile = os.path.join(jacpath, 'nfov_jac.fits')
    #     cstratfile = os.path.join(modelpath, 'cstrat_spec_band3_130deg.yaml')

    #     paramsetupfile = os.path.join(modelpath, 'homf_params_spec_band3_130deg.yaml')
    #     params = loadyaml(paramsetupfile)
    #     nrow = params['n_excam']
    #     ncol = nrow
    
    # elif mode == 'wfov_band4_360deg':
    #     modelpath = os.path.join(howfscpath, 'model', 'testdata', 'wfov_band4_360deg')
    #     cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
    #     jacfile = os.path.join(jacpath, 'nfov_jac.fits')
    #     cstratfile = os.path.join(modelpath, 'cstrat_wfov_band4_360deg.yaml')

    #     paramsetupfile = os.path.join(modelpath, 'homf_params_wfov_band4_360deg.yaml')
    #     params = loadyaml(paramsetupfile)
    #     nrow = params['n_excam']
    #     ncol = nrow

    # else:
    #     raise ValueError('Invalid coronagraph mode type')

    lrow = nclean//2 - nrow//2
    lcol = nclean//2 - ncol//2

    # cfg
    cfg = CoronagraphMode(cfgfile)

    nlam = len(cfg.sl_list)
    ndm = 1

    dm10 = cfg.initmaps[0]
    dm20 = cfg.initmaps[1]

    # cstratfile
    cstrat = ControlStrategy(cstratfile)

    # nrow, ncol, croplist
    croplist = [(lrow, lcol, nrow, ncol)]*(nlam*ndm)
    subcroplist = [(lrow, lcol, nrow, ncol)]*(nlam)

    # jac, jtwj_map, n2clist
    if args.precomp in ['precomp_all_once']:
        t0 = time.time()
        jac, jtwj_map, n2clist = howfsc_precomputation(
            cfg=cfg,
            dmset_list=[dm10, dm20],
            cstrat=cstrat,
            subcroplist=subcroplist,
            jacmethod='fast',
            do_n2clist=True,
            num_process=args.num_process,
            num_threads=args.num_threads,
        )
        t1 = time.time()
        print('Jac/JTWJ/n2clist computation time: ' + str(t1-t0) + ' seconds')
        
    elif args.precomp in ['precomp_jacs_once', 'precomp_jacs_always']:
        t0 = time.time()
        jac, jtwj_map, _ = howfsc_precomputation(
            cfg=cfg,
            dmset_list=[dm10, dm20],
            cstrat=cstrat,
            subcroplist=subcroplist,
            jacmethod='fast',
            do_n2clist=False,
            num_process=args.num_process,
            num_threads=args.num_threads,
        )
        t1 = time.time()
        print('Initial jac calc time: ' + str(t1-t0) + ' seconds')

    else: # load_all
        jac = fits.getdata(jacfile)
        jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)

    abs_dm1 = dm10
    abs_dm2 = dm20
    ni_list = []
    beta_list = []
    dmmultgain_list = []

    for iteration in range(1, niter+1): # var is number of next iteration

        t0 = time.time()
        abs_dm1, abs_dm2, dh_cube, next_c, prev_c, beta, dmmultgain = efc_computation(
            abs_dm1, abs_dm2, cfg, jac, jtwj_map, cstrat, croplist, iteration)
        t1 = time.time()

        abs_dm1list.append(abs_dm1)
        abs_dm2list.append(abs_dm2)
        ni_list.append(next_c)
        beta_list.append(beta)
        dmmultgain_list.append(dmmultgain)

        # Dump intermediate files.
        fits.writeto(
            os.path.join(out_path, f'iter_{iteration:03d}_dm1.fits'),
            abs_dm1, overwrite=True)
        fits.writeto(
            os.path.join(out_path, f'iter_{iteration:03d}_dm2.fits'),
            abs_dm2, overwrite=True)
        with open(os.path.join(out_path, f'iter_{iteration:03d}_ni.txt'), 'w') as f:
            f.write(f'{next_c:.4e}')

        if show_plot:
            plt.figure(1)
            plt.clf()
            plt.imshow(abs_dm1)
            plt.title('DM1 V')
            plt.set_cmap('viridis')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.pause(0.1)

            plt.figure(2)
            plt.clf()
            plt.imshow(abs_dm2)
            plt.title('DM2 V')
            plt.set_cmap('viridis')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.pause(0.1)
            
            plt.figure(3)
            plt.clf()
            img = plt.imshow(np.log10(np.mean(dh_cube, axis=0)))
            img.set_clim(-9, -3)
            plt.title('Mean NI: %.2e' % next_c)
            plt.set_cmap('magma')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.pause(0.1)   

        print('-----------------------------------')
        print('Iteration: %d' % iteration)
        print('EFC computation time: %.2f seconds' % (t1-t0))
        print('Previous contrast: %.3e' % (prev_c))
        print('Next contrast:     %.3e' % (next_c))
        print('Next beta: %.2f' % (beta))
        print('Next dmmultgain: %.2f' % (dmmultgain))

        # Skip the very last Jacobian that never gets used
        if args.precomp in ['precomp_jacs_always'] and iteration < niter:
            t0 = time.time()
            jac, jtwj_map, _ = howfsc_precomputation(
                cfg=cfg,
                dmset_list=[abs_dm1, abs_dm2],
                cstrat=cstrat,
                subcroplist=subcroplist,
                jacmethod='fast',
                do_n2clist=False,
                num_process=args.num_process,
                num_threads=args.num_threads,
            )
            t1 = time.time()
            print('Jac recalc time: %.2f seconds' % (t1-t0))

    # Write out final values
    fits.writeto(
        os.path.join(out_path, f'iter_{iteration:03d}_dh_cube.fits'),
        dh_cube, overwrite=True)
    fits.writeto(
        os.path.join(out_path, f'history_ni.fits'),
        np.asarray(ni_list), overwrite=True)
    fits.writeto(
        os.path.join(out_path, f'history_beta.fits'),
        np.asarray(beta_list), overwrite=True)
    fits.writeto(
        os.path.join(out_path, f'history_dmmultgain.fits'),
        np.asarray(dmmultgain_list), overwrite=True)
