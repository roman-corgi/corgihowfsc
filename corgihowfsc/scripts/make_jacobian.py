import argparse
import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

from astropy.io import fits

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from howfsc.control.calcjacs import calcjacs
from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check

VALID_JACMETHODS = ['normal', 'fast']
DEFAULT_OUTPUT_SUBDIR = 'corgiloop_data/jacobians'


def parse_args():
    ap = argparse.ArgumentParser(
        prog='python make_jacobian.py',
        description=(
            'Generate and save a Jacobian FITS file using the local '
            'corgihowfsc model setup.\n\n'
            'Example:\n'
            '  python make_jacobian.py --mode nfov_band1 --dark_hole 360deg '
            '--jacmethod fast --num_process 0 --num_threads 1\n\n'
            'By default, output is written under ~/corgiloop_data/jacobians/.\n'
            'By default, the script uses the mode-specific DM start maps returned '
            'by corgihowfsc.utils.howfsc_initialization.load_files(...).\n'
            'Use --dm1_start and --dm2_start together to override that default '
            'starting point when you want to linearize the Jacobian around a '
            'different DM state.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument('--mode', default='nfov_band1', type=str,
                    help='Corgihowfsc model family, e.g. nfov_band1, spec_band2, spec_band3, wfov_band4.')
    ap.add_argument('--dark_hole', default='360deg', type=str,
                    help='Dark-hole configuration inside the selected mode.')
    ap.add_argument('--jacmethod', default='fast', type=str,
                    choices=VALID_JACMETHODS,
                    help='Jacobian calculation method.')
    ap.add_argument('--num_process', default=0, type=int,
                    help='Number of Jacobian worker processes. Use 0 for half the available CPUs.')
    ap.add_argument('--num_threads', default=None, type=int,
                    help='MKL threads per process. If omitted, follows HOWFSC defaults.')
    ap.add_argument('--base_path', default=str(Path.home()), type=str,
                    help='Root path under which Jacobians will be saved.')
    ap.add_argument('--output', default=None, type=str,
                    help='Explicit output FITS path. Overrides the timestamped default location.')
    ap.add_argument('--dm1_start', default=None, type=str,
                    help='Optional DM1 start-map filename relative to the selected model path, or an absolute path.')
    ap.add_argument('--dm2_start', default=None, type=str,
                    help='Optional DM2 start-map filename relative to the selected model path, or an absolute path.')
    return ap.parse_args()


def set_num_processes(num_process):
    if num_process is None:
        num_process = int(os.environ.get('HOWFS_CALCJAC_NUM_PROCESS', 1))

    check.nonnegative_scalar_integer(num_process, 'num_process', TypeError)
    if num_process == 0:
        num_process = multiprocessing.cpu_count() // 2
    return num_process


def set_num_threads(num_process, num_threads):
    if num_threads is None:
        num_threads = os.environ.get('HOWFS_CALCJAC_NUM_THREADS')
        if num_threads is None and os.environ.get('MKL_NUM_THREADS') is None \
           and num_process > 1:
            num_threads = 1

    if num_threads is not None:
        if isinstance(num_threads, str):
            num_threads = int(num_threads)
        check.positive_scalar_integer(num_threads, 'num_threads', TypeError)

    return num_threads

def calculate_jacobian(cfgfile, output, jacmethod,
                       num_process=None, num_threads=None):
    """

    A wrapper function around `caljacs` to handle input validation, parallelism settings, and output file writing for Jacobian calculation. It will calculate a Jacobian and write it to a FITS file.

    The number of actuators is derived automatically from the model — all
    actuators across every DM defined in the config file are included.

    Args:
        cfgfile (str): Path to ``howfsc_optical_model.yaml``.
        output (str): Destination path for the output FITS file.
        jacmethod (str): Calculation method — 'normal' or 'fast'.
        num_process (int or None): Worker processes (see
            :func:`set_num_processes` for rules).
        num_threads (int or None): MKL threads per process (see
            :func:`set_num_threads` for rules).
    """
    check.string(jacmethod, 'jacmethod', TypeError)
    if jacmethod not in VALID_JACMETHODS:
        raise ValueError(
            f"Invalid jacmethod '{jacmethod}'. Valid options: {VALID_JACMETHODS}"
        )

    cfg = CoronagraphMode(cfgfile)

    # Derive the total actuator count by summing nact^2 across all DMs.
    # nact is the number of actuators along one axis of a square DM grid,
    # so the total per DM is nact^2, and we sum over all DMs in cfg.dmlist.
    nact = int(np.cumsum(
        [0] + [d.registration['nact'] ** 2 for d in cfg.dmlist]
    )[-1])
    ijlist = range(nact)  # one entry per actuator, iterated by calcjacs

    # dm0list=None tells calcjacs to use the initial DM map built into the cfg
    # rather than a custom starting state
    dm0list = None

    # Resolve parallelism settings from arguments / environment / defaults
    num_process = set_num_processes(num_process)
    num_threads = set_num_threads(num_process, num_threads)

    # Pin MKL thread count if needed, saving any pre-existing value so we can
    # restore it exactly after the calculation (important when this function is
    # called from within a larger script that may have its own MKL settings)
    saved_mkl = None
    if num_threads is not None:
        saved_mkl = os.environ.get('MKL_NUM_THREADS')
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        print(f'Set MKL_NUM_THREADS = {num_threads}')
    elif 'MKL_NUM_THREADS' in os.environ:
        # Already set externally. Leave it alone and just report it
        print(f"MKL_NUM_THREADS = {os.environ['MKL_NUM_THREADS']} (from environment)")

    print(f'Beginning Jacobian calculation: {num_process} process(es), method={jacmethod}')
    print(f'Total actuators: {nact}')
    t0 = time.time()

    try:
        jac = calcjacs(cfg, ijlist, dm0list, jacmethod=jacmethod,
                       num_process=num_process)
    finally:
        # Always restore MKL_NUM_THREADS even if calcjacs raises, so the
        # calling environment is not left in an unexpected state
        if num_threads is not None:
            if saved_mkl is not None:
                os.environ['MKL_NUM_THREADS'] = saved_mkl
            else:
                del os.environ['MKL_NUM_THREADS']
            print('Restored MKL_NUM_THREADS to previous value')

    print(f'Jacobian calculation complete: {time.time() - t0:.2f} seconds')

    if output is not None:
        print(f'Writing Jacobian to: {output}')
        fits.writeto(output, jac)

if __name__ == '__main__':
    main()