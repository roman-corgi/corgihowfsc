import time
import os
import argparse
import cProfile
import pstats
import logging

import numpy as np
import astropy.io.fits as pyfits


def get_args_jupyter(niter=5,
                    mode='narrowfov',
                    profile=False,
                    fracbadpix=0,
                    nbadpacket=0,
                    nbadframe=0,
                    logfile=None,
                    precomp='load_all',
                    num_process=None,
                    num_threads=None,
                    fileout=None,
                    stellarvmag=None,
                    stellartype=None,
                    stellarvmagtarget=None,
                    stellartypetarget=None,
                    jacpath=None):
        """
        Initialize HOWFSC simulation with all required variables and configurations.
        Returns all variables needed for the main simulation loop.
        """

        # Copy the path setup from original script
        # eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
        # howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
        #
        # defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')

        # Create args object with all the original parameters
        class Args:
            pass

        args = Args()
        args.niter = niter
        args.mode = mode
        args.profile = profile
        args.fracbadpix = fracbadpix
        args.nbadpacket = nbadpacket
        args.nbadframe = nbadframe
        args.logfile = logfile
        args.precomp = precomp
        args.num_process = num_process
        args.num_threads = num_threads
        args.fileout = fileout
        args.stellarvmag = stellarvmag
        args.stellartype = stellartype
        args.stellarvmagtarget = stellarvmagtarget
        args.stellartypetarget = stellartypetarget
        args.jacpath = jacpath

        return args
def get_args(defjacpath):
    '''
    Note: this cannot be run from jupyter notebook

    :param defjacpath:
    :return:
    '''
    # setup for cmd line args
    ap = argparse.ArgumentParser(prog='python nulltest_gitl.py',
                                 description="Run a nulling sequence, using the optical model as the data source.  Outputs will be displayed to the command line.")
    ap.add_argument('-n', '--niter', default=5, help="Number of iterations to run.  Defaults to 5.", type=int)
    ap.add_argument('--mode', default='widefov',
                    choices=['widefov', 'narrowfov', 'spectroscopy', 'nfov_dm', 'nfov_flat'],
                    help="coronagraph mode from test data; must be one of 'widefov', 'narrowfov', 'nfov_dm', 'nfov_flat', or 'spectroscopy'.  Defaults to 'widefov'.")
    ap.add_argument('--profile', action='store_true',
                    help='If present, runs the Python cProfile profiler on howfsc_computation and displays the top 20 howfsc/ contributors to cumulative time')
    ap.add_argument('-p', '--fracbadpix', default=0, type=float,
                    help="Fraction of pixels, in [0, 1], to make randomly bad (e.g. due to cosmic rays).  Defaults to 0, with no bad pixels.")
    ap.add_argument('--nbadpacket', default=0, type=int,
                    help="Number of GITL packets (3 rows x 153 cols) irrecoverably lost due to errors, to be replaced by NaNs.  Defaults to 0.  Same number is applied to every iteration, although not in the same place.")
    ap.add_argument('--nbadframe', default=0, type=int,
                    help="Number of entire GITL frames (153 x 153) irrecoverably lost due to errors, to be replaced by NaNs.  Defaults to 0.  Same number is applied to every iteration, although not in the same place.")
    ap.add_argument('--logfile', default=None, help="If present, absolute path to file location to log to.")
    ap.add_argument('--precomp', default='load_all',
                    choices=['load_all', 'precomp_all_once', 'precomp_jacs_once', 'precomp_jacs_always'],
                    help="Specifies Jacobian precomputation behavior.  'load_all' will load everything from files at the start and leave them fixed.  'precomp_all_once' will compute a Jacobian, JTWJs, and n2clist once at start.  'precomp_jacs_once' will compute a Jacobian and JTWJs once at start, and load n2clist from file.  'precomp_jacs_always' will compute a Jacobian and JTWJs at start and at every iteration; n2clist will be loaded from file.")

    ap.add_argument(
        '--num_process', default=None,
        help="number of processes, defaults to None, indicating no multiprocessing. If 0, then howfsc_precomputation() defaults to the available number of cores",
        type=int
    )
    ap.add_argument(
        '--num_threads', default=None,
        help="set mkl_num_threads to use for parallel processes for calcjacs(). If None (default), then do nothing (number of threads might also be set externally through environment variable 'MKL_NUM_THREADS' or 'HOWFS_CALCJAC_NUM_THREADS'",
        type=int
    )

    ap.add_argument('--fileout', default=None,
                    help="If present, absolute path to file location of output .fits (including \'.fits\' at the end) file containing framelist and prev_exptime_list, as well as nlam stored as a header.")
    ap.add_argument('--stellarvmag', default=None, type=float,
                    help='If present, magnitude of the reference star desired will be updated in the hconf file (for parameter stellar_vmag in hconf file).')
    ap.add_argument('--stellartype', default=None,
                    help='If present, type of the reference star desired will be updated in the hconf file (for parameter stellar_type in hconf file).')
    ap.add_argument('--stellarvmagtarget', default=None, type=float,
                    help='If present, magnitude of the target star desired will be updated in the hconf file (for parameter stellar_vmag_target in hconf file).')
    ap.add_argument('--stellartypetarget', default=None,
                    help='If present, type of the target star desired will be updated in the hconf file (for parameter stellar_type_target in hconf file).')

    ap.add_argument('-j', '--jacpath', default=defjacpath, help="absolute path to read Jacobian files from", type=str)
    args = ap.parse_args()

    return args



def load_files(args, howfscpath):
    # User params
    mode = args.mode
    isprof = args.profile
    logfile = args.logfile
    nbadpacket = args.nbadpacket
    nbadframe = args.nbadframe

    jacpath = args.jacpath
    if nbadpacket < 0:
        raise ValueError('Number of bad packets cannot be less than 0.')
    if nbadframe < 0:
        raise ValueError('Number of bad frames cannot be less than 0.')

    if isprof:
        pr = cProfile.Profile()
        pass

    # Set up logging
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
        pass
    else:
        logging.basicConfig(level=logging.INFO)
        pass
    log = logging.getLogger(__name__)

    exptime = 10  # FIXME this should be derived from contrast eventually
    contrast = 1e-5  # "starting" value to bootstrap getting we0

    if mode == 'nfov_dm':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgfile = os.path.join(modelpath, 'narrowfov_dm.yaml')
        jacfile = os.path.join(jacpath, 'nfov_dm_jac_full.fits')
        cstratfile = os.path.join(modelpath, 'cstrat_nfov_dm.yaml')
        probe0file = os.path.join(modelpath,
                                  'nfov_dm_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(modelpath,
                                  'nfov_dm_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(modelpath,
                                  'nfov_dm_dmrel_1.0e-05_sinud.fits')
        hconffile = os.path.join(modelpath, 'hconf_nfov_dm.yaml')
        n2clistfiles = [
            os.path.join(modelpath, 'nfov_dm_n2c_idx0.fits'),
            os.path.join(modelpath, 'nfov_dm_n2c_idx1.fits'),
            os.path.join(modelpath, 'nfov_dm_n2c_idx2.fits'),
        ]
        pass
    elif mode == 'nfov_flat':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgfile = os.path.join(modelpath, 'narrowfov_flat.yaml')
        jacfile = os.path.join(jacpath, 'nfov_flat_jac_full.fits')
        cstratfile = os.path.join(modelpath, 'cstrat_nfov_flat.yaml')
        probe0file = os.path.join(modelpath,
                                  'nfov_flat_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(modelpath,
                                  'nfov_flat_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(modelpath,
                                  'nfov_flat_dmrel_1.0e-05_sinud.fits')
        hconffile = os.path.join(modelpath, 'hconf_nfov_flat.yaml')
        n2clistfiles = [
            os.path.join(modelpath, 'nfov_flat_n2c_idx0.fits'),
            os.path.join(modelpath, 'nfov_flat_n2c_idx1.fits'),
            os.path.join(modelpath, 'nfov_flat_n2c_idx2.fits'),
        ]
        pass
    elif mode == 'narrowfov':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgfile = os.path.join(modelpath, 'narrowfov.yaml')
        jacfile = os.path.join(jacpath, 'narrowfov_jac_full.fits')
        cstratfile = os.path.join(modelpath, 'cstrat_narrowfov.yaml')
        probe0file = os.path.join(modelpath,
                                  'narrowfov_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(modelpath,
                                  'narrowfov_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(modelpath,
                                  'narrowfov_dmrel_1.0e-05_sinud.fits')
        hconffile = os.path.join(modelpath, 'hconf_narrowfov.yaml')
        n2clistfiles = [
            os.path.join(modelpath, 'narrowfov_n2c_idx0.fits'),
            os.path.join(modelpath, 'narrowfov_n2c_idx1.fits'),
            os.path.join(modelpath, 'narrowfov_n2c_idx2.fits'),
        ]
        pass

    else:
        # should not reach here; argparse should catch this
        raise ValueError('Invalid coronagraph mode type')


    return modelpath, cfgfile, jacfile, cstratfile, probe0file, probe1file, probe2file, hconffile, n2clistfiles
