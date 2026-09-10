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
                    help=(
                        'Optional DM1 start-map override. Must be used together '
                        'with --dm2_start. May be either a filename relative to '
                        'the selected model directory or an absolute path. '
                        'Use this when you want the Jacobian computed about a '
                        'different DM operating point than the default start map.'
                    ))
    ap.add_argument('--dm2_start', default=None, type=str,
                    help=(
                        'Optional DM2 start-map override. Must be used together '
                        'with --dm1_start. May be either a filename relative to '
                        'the selected model directory or an absolute path.'
                    ))
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


def default_output_path(base_path, mode, dark_hole):
    outdir = Path(base_path) / DEFAULT_OUTPUT_SUBDIR
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return str(outdir / f'jacobian_{mode}_{dark_hole}_{timestamp}.fits')


def build_cfg_and_dmset(mode, dark_hole, dmstartmap_filenames):
    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
        dmstartmap_filenames=dmstartmap_filenames,
    )
    _modelpath, cfgfile, _jacfile, _cstratfile, _probefiles, _hconffile, _n2clistfiles, dmstartmaps = load_files(args, howfscpath)

    cfg = CoronagraphMode(cfgfile)
    dm10, dm20 = dmstartmaps
    return cfg, [dm10, dm20]


def calculate_jacobian(cfg, dmset_list, jacmethod, num_process=None, num_threads=None):
    check.string(jacmethod, 'jacmethod', TypeError)
    if jacmethod not in VALID_JACMETHODS:
        raise ValueError(
            f"Invalid jacmethod '{jacmethod}'. Valid options: {VALID_JACMETHODS}"
        )

    ndmact = sum(d.registration['nact'] ** 2 for d in cfg.dmlist)
    ijlist = range(ndmact)

    num_process = set_num_processes(num_process)
    num_threads = set_num_threads(num_process, num_threads)

    saved_mkl = None
    if num_threads is not None:
        saved_mkl = os.environ.get('MKL_NUM_THREADS')
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        logging.info('Set MKL_NUM_THREADS=%s', num_threads)
    elif 'MKL_NUM_THREADS' in os.environ:
        logging.info('Using existing MKL_NUM_THREADS=%s',
                     os.environ['MKL_NUM_THREADS'])

    logging.info('Beginning Jacobian calculation')
    logging.info('jacmethod=%s, num_process=%s, total_actuators=%s',
                 jacmethod, num_process, ndmact)
    t0 = time.time()

    try:
        jac = calcjacs(cfg, ijlist, dmset_list, jacmethod=jacmethod,
                       num_process=num_process)
    finally:
        if num_threads is not None:
            if saved_mkl is None:
                del os.environ['MKL_NUM_THREADS']
            else:
                os.environ['MKL_NUM_THREADS'] = saved_mkl

    logging.info('Jacobian calculation complete in %.2f s', time.time() - t0)
    return jac


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dmstartmap_filenames = None
    if args.dm1_start or args.dm2_start:
        if not (args.dm1_start and args.dm2_start):
            raise ValueError('Both --dm1_start and --dm2_start must be provided together.')
        dmstartmap_filenames = [args.dm1_start, args.dm2_start]

    cfg, dmset_list = build_cfg_and_dmset(
        mode=args.mode,
        dark_hole=args.dark_hole,
        dmstartmap_filenames=dmstartmap_filenames,
    )

    output = args.output or default_output_path(
        args.base_path, args.mode, args.dark_hole
    )
    output_parent = Path(output).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)

    jac = calculate_jacobian(
        cfg=cfg,
        dmset_list=dmset_list,
        jacmethod=args.jacmethod,
        num_process=args.num_process,
        num_threads=args.num_threads,
    )
    fits.writeto(output, jac, overwrite=True)
    logging.info('Wrote Jacobian to %s', output)


if __name__ == '__main__':
    main()
