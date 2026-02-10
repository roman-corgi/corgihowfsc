import time
import os
import argparse
import cProfile
import pstats
import logging
import glob

import numpy as np
from astropy.io import fits

from howfsc.util.load import load

def get_args(niter=5,
                    mode='narrowfov',
                    dark_hole='360deg',
                    probe_shape='default',
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
                    jacpath=None,
                    dmstartmap_filenames=None):
        """
        Initialize HOWFSC simulation with all required variables and configurations.
        Returns all variables needed for the main simulation loop.

        niter : int, optional
            Number of iterations to run.  Defaults to 5.
        mode : str, optional
            Coronagraph mode from test data; must be one of 'widefov', 'narrowfov',
            'nfov_dm', 'nfov_flat', or 'spectroscopy'.  Defaults to 'narrowfov'.
        isprof : bool, optional
            If True, runs the Python cProfile profiler on howfsc_computation and
            displays the top 20 howfsc/ contributors to cumulative time.
        logfile : str, optional
            If present, absolute path to file location to log to.
        fracbadpix : float, optional
            Fraction of pixels, in [0, 1], to make randomly bad (e.g. due to cosmic
            rays).  Defaults to 0, with no bad pixels.
        nbadpacket : int, optional
            Number of GITL packets (3 rows x 153 cols) irrecoverably lost due to
            errors, to be replaced by NaNs. Same number is applied to every iteration,
            although not in the same place. Defaults to 0.
        nbadframe : int, optional
            Number of entire GITL frames (153 x 153) irrecoverably lost due to
            errors, to be replaced by NaNs. Same number is applied to every iteration,
            although not in the same place. Defaults to 0.
        fileout : str, optional
            If present, absolute path to file location (including .fits at the end) to
            write output FITS file containing the final set of frames. prev_exptime_list,
            as well as nlam are stored as a header.
        stellar_vmag : float, optional
            If present, overrides the V-band magnitude of the star in the
            hconf file.
        stellar_type : str, optional
            If present, overrides the stellar type of the star in the hconf file for
            the mode.
        stellar_vmag_target : float, optional
            If present, overrides the V-band magnitude of the target in the
            hconf file.
        stellar_type_target : str, optional
            If present, overrides the stellar type of the target in the hconf file.
        jacpath : str, optional
            Path to directory containing precomputed Jacobians. Expected file names
            are hard-coded for each coronagraph mode. Defaults to the jacdata directory
            in the howfsc repository.
        precomp : str, optional
            One of 'load_all', 'precomp_jacs_once', 'precomp_jacs_always',
            or 'precomp_all_once'.  This determines how the Jacobians and related
            data are handled.  Defaults to 'load_all'.
            'load_all' means that the Jacobians, JTWJ map, and n2clist are all
            loaded from files in jacpath and leaves them fixed throughout a loop.
            'precomp_jacs_once' means that the Jacobians and JTWJ map are computed
            once at the start of the sequence, and the n2clist is loaded from files
            in jacpath.
            'precomp_jacs_always' means that the Jacobians and JTWJ map are computed
            at the start of the sequence and then recomputed at the start of every
            iteration except the last one; the n2clist is loaded from files in jacpath.
            'precomp_all_once' means that the Jacobians, JTWJ map, and n2clist are
            all computed once at the start of the sequence.
        num_process : int, optional
            Number of processes to use for Jacobian computation.  If None (the
            default), no multiprocessing is used.  If 0, uses half the number of CPU
            cores on the machine.
        num_threads : int, optional
            Sets mkl_num_threads to this value for parallel processing of calcjacs().
            If None (the default), the environment variable MKL_NUM_THREADS or
            HOWFS_CALCJAC_NUM_THREADS is used if it exists; otherwise, it does nothing.
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
        args.dark_hole = dark_hole
        args.probe_shape = probe_shape
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
        args.dmstartmap_filenames = dmstartmap_filenames
        # args.dm_start_shape = dm_start_shape

        return args

def get_args_cmd(defjacpath):
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
    ap.add_argument('--probe_shape', default='default', choices=['default', 'single'],
                    help="Shape of the probes: 'default' (Sinc) or 'single' (Single Actuator).")
    args = ap.parse_args()

    return args



def load_files(args, howfscpath):
    # User params
    mode = args.mode
    isprof = args.profile
    logfile = args.logfile
    nbadpacket = args.nbadpacket
    nbadframe = args.nbadframe
    dmstartmap_filenames = args.dmstartmap_filenames

    jacpath = args.jacpath
    if nbadpacket < 0:
        raise ValueError('Number of bad packets cannot be less than 0.')
    if nbadframe < 0:
        raise ValueError('Number of bad frames cannot be less than 0.')

    if isprof:
        pr = cProfile.Profile()
        pass

    # Check probes shapes : Default = sinc-sin-sin, others are alternates probes
    supported_shapes = {'default', 'single', 'gaussian', 'unmodulated_sinc'}

    if args.probe_shape in supported_shapes:
        pass
    else:
        raise ValueError(
            f"Probe shape '{args.probe_shape}' not recognized. "
            f"Expected one of: {', '.join(supported_shapes)}"
        )

    # Set up logging
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
        pass
    else:
        logging.basicConfig(level=logging.INFO)
        pass
    log = logging.getLogger(__name__)

    model_path_all = os.path.join(howfscpath, 'model', 'every_mask_config')
    n2clistfiles = [
        os.path.join(model_path_all, 'ones_like_fs.fits'),
        os.path.join(model_path_all, 'ones_like_fs.fits'),
        os.path.join(model_path_all, 'ones_like_fs.fits'),
    ]
    if mode == 'nfov_band1':
        modelpath_band = os.path.join(howfscpath, 'model', 'nfov_band1')
        modelpath = os.path.join(modelpath_band, mode+'_'+args.dark_hole)
        probepath = os.path.join(howfscpath, 'model', 'probes')
        cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
        if jacpath is not None:
            jacfile = os.path.join(jacpath, 'jac' + mode + '_' + args.dark_hole + '.fits')
        else:
            jacfile = []

        if '360deg' in args.dark_hole:
            hconffile = os.path.join(modelpath_band, 'hconf_nfov_flat.yaml')
            cstratfile = os.path.join(modelpath, 'cstrat_nfov_band1.yaml')
            if args.probe_shape == 'default':
                # Sinc-sin-sin probes
                probe0file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_cos.fits')
                probe1file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinlr.fits')
                probe2file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinud.fits')
            elif args.probe_shape == 'single':
                # Single actuator probes
                probe0file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act0.fits')
                probe1file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act1.fits')
                probe2file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act2.fits')
            elif args.probe_shape == 'gaussian':
                # Gaussian probes
                probe0file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian0.fits')
                probe1file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian1.fits')
                probe2file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian2.fits')
            elif args.probe_shape == 'unmodulated_sinc':
                # Unmodulated sinc probes
                probe0file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinc.fits')
                probe1file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinc_shifted_right.fits')
                probe2file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinc_shifted_diag_ur.fits')
            else:
                # Raise an error if the probe shape is not recognized
                raise ValueError(f"Probe shape '{args.probe_shape}' is not recognized. "
                                 "Supported shapes are: 'default', 'single', 'gaussian' and 'unmodulated_sinc'.")
            # if args.dm_start_shape is not None:
            #     dm_start_file = os.path.join(modelpath, args.dm_start_shape)
            # else:
            #     # If no starting point is given, we will load the last file globbed
            #     start_options = glob.glob(os.path.join(modelpath, 'iter_*_dm*'))
            #     start_parts = start_options[0].split('\\')[-1].split('_')
            #     dm_start_file = os.path.join(modelpath, start_parts[0] + '_' + start_parts[1] + '_')
            #     print('Using ' + start_parts[0] + '_' + start_parts[1] + '_' + ' as starting DM shape')

            if dmstartmap_filenames is None:
                dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']

        elif 'half' in args.dark_hole:
            hconffile = os.path.join(modelpath_band, 'hconf_nfov_flat.yaml')
            cstratfile = os.path.join(modelpath, 'cstrat_nfov_band1_half.yaml')

            if args.probe_shape == 'single':
                # Single actuator alternate probes
                probe0file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act0.fits')
                probe1file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act1.fits')
                probe2file = os.path.join(probepath, 'narrowfov_dmrel_1.0e-05_act2.fits')
            elif args.probe_shape == 'default':
                # Sinc probes
                probe0file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_cos.fits')
                probe1file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinlr.fits')
                probe2file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_sinud.fits')
            elif args.probe_shape == 'gaussian':
                # Gaussian alternate probes
                probe0file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian0.fits')
                probe1file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian1.fits')
                probe2file = os.path.join(probepath, 'nfov_dm_dmrel_4_1.0e-05_gaussian2.fits')
            else:
                # Raise an error if the probe shape is not recognized
                raise ValueError(f"Probe shape '{args.probe_shape}' is not recognized. "
                                 "Supported shapes are: 'single', 'default' and 'gaussian'.")
            if 'top' in args.dark_hole:
                if dmstartmap_filenames is None:
                    dmstartmap_filenames = ['iter_061_dm1.fits', 'iter_061_dm2.fits']

        probefiles = {}
        probefiles[0] = probe0file
        probefiles[2] = probe1file
        probefiles[1] = probe2file

    elif mode == 'spec_band2':
        modelpath_band = os.path.join(howfscpath, 'model', 'spec_band2')
        modelpath = os.path.join(modelpath_band, mode + '_' + args.dark_hole)
        probepath = os.path.join(howfscpath, 'model', 'probes')

        hconffile = os.path.join(modelpath_band, 'hconf_spec_band2.yaml')

        cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
        cstratfile = os.path.join(modelpath, 'cstrat_spec_band2.yaml')

        probe0file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_sinud.fits')
        probefiles = {}
        probefiles[0] = probe0file
        probefiles[2] = probe1file
        probefiles[1] = probe2file

        if jacpath is not None:
            jacfile = os.path.join(jacpath, 'spec_band2_jac.fits')
        else:
            jacfile = []

        # TODO: check how many subband folders there are and load the appropriate number here
        n2clistfiles = [
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
        ]

    elif mode == 'spec_band3':
        modelpath_band = os.path.join(howfscpath, 'model', 'spec_band3')
        modelpath = os.path.join(modelpath_band, mode + '_' + args.dark_hole)
        probepath = os.path.join(howfscpath, 'model', 'probes')

        hconffile = os.path.join(modelpath_band, 'hconf_spec_band3.yaml')

        cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
        cstratfile = os.path.join(modelpath, 'cstrat_spec_band3.yaml')

        probe0file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(probepath, 'spectroscopy_dmrel_1.0e-05_sinud.fits')
        probefiles = {}
        probefiles[0] = probe0file
        probefiles[2] = probe1file
        probefiles[1] = probe2file

        if jacpath is not None:
            jacfile = os.path.join(jacpath, 'cstrat_spec_band3.fits')
        else:
            jacfile = []

        n2clistfiles = [
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
        ]

        if dmstartmap_filenames is None:
            dmstartmap_filenames = ['iter_061_dm1.fits', 'iter_061_dm2.fits']

    elif mode == 'wfov_band4':
        modelpath_band = os.path.join(howfscpath, 'model', 'wfov_band4')
        modelpath = os.path.join(modelpath_band, mode + '_' + args.dark_hole)
        probepath = os.path.join(howfscpath, 'model', 'probes')

        hconffile = os.path.join(modelpath_band, 'hconf_wfov_band4.yaml')

        cfgfile = os.path.join(modelpath, 'howfsc_optical_model.yaml')
        cstratfile = os.path.join(modelpath, 'cstrat_wfov_band4.yaml')

        probe0file = os.path.join(probepath, 'wfov_dmrel_1e-5_cos_constrained.fits')
        probe1file = os.path.join(probepath, 'wfov_dmrel_1e-5_sinlr_constrained.fits')
        probe2file = os.path.join(probepath, 'wfov_dmrel_1e-5_sinud_constrained.fits')
        probefiles = {}
        probefiles[0] = probe0file
        probefiles[2] = probe1file
        probefiles[1] = probe2file

        if jacpath is not None:
            jacfile = os.path.join(jacpath, 'wfov_band4_jac.fits')
        else:
            jacfile = []

        n2clistfiles = [
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
            os.path.join(model_path_all, 'ones_like_fs.fits'),
        ]

        if dmstartmap_filenames is None:
            dmstartmap_filenames = ['iter_061_dm1.fits', 'iter_061_dm2.fits']

    else:
        # should not reach here; argparse should catch this
        raise ValueError('Invalid coronagraph mode type')

    dmstartmaps = [
        fits.getdata(os.path.join(modelpath, dmstartmap_filenames[0])),
        fits.getdata(os.path.join(modelpath, dmstartmap_filenames[1])),
    ]
    # dmstartmaps = load_dm_start_maps(dm_start_file)


    return modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps


# def load_dm_start_maps(dm_start_file):
#     dmkeylist = ['DM1', 'DM2']
#     # Load DM settings used to collect channel data
#     dmstartmaps = []
#     for dmkey in dmkeylist:
#         ipath = dm_start_file + dmkey.lower() + '.fits'
#         dmstartmap = load(ipath)
#         dmstartmaps.append(dmstartmap)

#     return dmstartmaps




