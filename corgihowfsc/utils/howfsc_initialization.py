import time
import os
import argparse
import cProfile
import pstats
import logging

import numpy as np
import astropy.io.fits as pyfits

def get_args(defjacpath):
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
