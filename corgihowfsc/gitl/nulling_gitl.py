
# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#pylint: disable=line-too-long
"""
ugly but functional test of howfsc_computation
"""

import time
import os
import argparse
import cProfile
import pstats
import io

import logging
log = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as pyfits

import eetc
from eetc.cgi_eetc import CGIEETC

import howfsc
from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjtwj import JTWJMap

from howfsc.model.mode import CoronagraphMode

from howfsc.util.loadyaml import loadyaml
from howfsc.util.gitl_tools import param_order_to_list
from howfsc.precomp import howfsc_precomputation

from corgihowfsc.gitl.modular_gitl import howfsc_computation
from corgihowfsc.utils.saving_output import save_outputs, save_outputs_iter
from corgihowfsc.utils.output_management import save_run_config, update_yml
from corgihowfsc.utils.gitl_worker import _collect_framelist
from corgihowfsc.gitl.gitl_funcs import get_initial_cam_params
from corgihowfsc.utils.metrics import get_ni


eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')


def nulling_gitl(cstrat, estimator, probes, normalization_strategy, imager, cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles, crop_params, dmstartmaps, metadata=None, output_every_iter=True, output_model_efield=True,parallel_approach=='robust'):
    """Run a nulling sequence, using the compact optical model as the data source.

    Parameters:
    -----------
    cstrat: a ControlStrategy object;
          this will be used to define the behavior
          of the wavefront control by setting the regularization, per-pixel
          weighting, multiplicative gain, and next-iteration probe height.  It will
          also contain information about fixed bad pixels.
    estimator: a Estimator object;
      this will be used to define the behavior
      of the wavefront estimator by determining the method to take the probe images and convert into an e-field estimate.

    probes: a Probe object;
        Defines the probing sequence for the Estimation scheme. Determines the number of probes, shape of probes, etc.

    normalization_strategy: a Normalization object;
        Defines the counts->contrast conversion including calculating the peakflux and performing the conversion.

    imager: a Imaging object;
        Defines the imaging behaviour, can use corgisim or compact model.

    cfg: a CoronagraphMode object;
        a CoronagraphMode object (i.e. optical model)

    args: a Arguments object;
        Defines loop parameters such as niter, filepaths, stellar properties, etc.

    probefiles: list of strings;
        List containing the probe file paths.

    hconffile: string;
        Path to hardware configuration.

    jacfile: string;
        Path to jacobian.
    modelpath, jacfile, probefiles, hconffile, n2clistfiles:
    """

    # User params
    niter = args.niter
    mode = args.mode
    isprof = args.profile
    logfile = args.logfile
    fracbadpix = args.fracbadpix
    nbadpacket = args.nbadpacket
    nbadframe = args.nbadframe
    fileout = args.fileout
    stellar_vmag = args.stellarvmag
    stellar_type = args.stellartype
    stellar_vmag_target = args.stellarvmagtarget
    stellar_type_target = args.stellartypetarget
    jacpath = args.jacpath
    precomp = args.precomp
    num_process = args.num_process
    num_threads = args.num_threads
    contrast = float(args.starting_contrast) # "starting" value to bootstrap getting we0

    safe_cpu_count = args.num_imager_worker # TODO - hard coding
    print('Using num_imager_worker = ', safe_cpu_count)

    # TODO - move this out from here and change it to os.sched_getaffinity
    if safe_cpu_count == None:
        safe_cpu_count = 1

    safe_cpu_count = args.num_imager_worker # TODO - hard coding
    print('Using num_imager_worker = ', safe_cpu_count)

    # TODO - move this out from here and change it to os.sched_getaffinity
    if safe_cpu_count == None:
        safe_cpu_count = 1

    # Make filout dir
    if args.fileout is not None:
        print('Making output directory: ', args.fileout)
        os.makedirs(os.path.dirname(args.fileout), exist_ok=True)

    # Set up logging
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
        pass
    else:
        logging.basicConfig(level=logging.INFO)
        pass

    log = logging.getLogger(__name__)

    config_path = save_run_config(args, args.fileout)
    log.info(f"Saved run configuration to {config_path}")

    update_yml(config_path, metadata)

    otherlist = []
    abs_dm1list = []
    abs_dm2list = []
    framelistlist = []
    scalelistout = []
    camlist = []

    # New lists compared to original
    measured_c = []
    pred_c = []
    ni_lists = {'ni_score': [], 'ni_inner': [], 'ni_outer': []}
    perfect_efield_list = []

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

    contrast = float(args.starting_contrast) # "starting" value to bootstrap getting we0

    # dm1_list, dm2
    # Get DM lists
    dm1_list, dm2_list, dmrel_list, dm10, dm20 = probes.get_dm_probes(cfg, probefiles, dmstartmaps)
    nlam = len(cfg.sl_list)
    ndm = 2 * len(dmrel_list) + 1
    nprobepair = len(dmrel_list)

    # cstratfile
    # cstrat = ControlStrategy(cstratfile)

    # nrow, ncol, croplist
    nrow = crop_params['nrow']
    ncol = crop_params['ncol']
    lrow = crop_params['lrow']
    lcol = crop_params['lcol']
    croplist = [(lrow, lcol, nrow, ncol)]*(nlam*ndm)
    subcroplist = [(lrow, lcol, nrow, ncol)]*(nlam)
    nrowperpacket = 3 # only used by packet-drop testing

    abs_dm1list.append(dm10)
    abs_dm2list.append(dm20)

    # jac, jtwj_map, n2clist
    print('Calculating jacobian and jtwj_map...')

    if precomp in ['precomp_all_once']:
        t0 = time.time()
        jac, jtwj_map, n2clist = howfsc_precomputation(
            cfg=cfg,
            dmset_list=[dm10, dm20],
            cstrat=cstrat,
            subcroplist=subcroplist,
            jacmethod='fast',
            do_n2clist=True,
            num_process=num_process,
            num_threads=num_threads,
        )
        t1 = time.time()
        log.info('Jac/JTWJ/n2clist computation time: ' + str(t1-t0) + ' seconds')
        pass
    elif precomp in ['precomp_jacs_once', 'precomp_jacs_always']:
        t0 = time.time()
        jac, jtwj_map, _ = howfsc_precomputation(
            cfg=cfg,
            dmset_list=[dm10, dm20],
            cstrat=cstrat,
            subcroplist=subcroplist,
            jacmethod='fast',
            do_n2clist=False,
            num_process=num_process,
            num_threads=num_threads,
        )
        t1 = time.time()
        log.info('Initial jac calc time: ' + str(t1-t0) + ' seconds')
        n2clist = []
        for n2cfn in n2clistfiles:
            n2clist.append(pyfits.getdata(n2cfn))
            pass
        pass
    else: # load_all
        jac = pyfits.getdata(jacfile)
        jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)
        n2clist = []
        for n2cfn in n2clistfiles:
            n2clist.append(pyfits.getdata(n2cfn))
            pass
        pass

    if stellar_vmag is not None:
        hconf['star']['stellar_vmag'] = stellar_vmag
    if stellar_type is not None:
        hconf['star']['stellar_type'] = stellar_type
    if stellar_vmag_target is not None:
        hconf['star']['stellar_vmag_target'] = stellar_vmag_target
    if stellar_type_target is not None:
        hconf['star']['stellar_type_target'] = stellar_type_target

    # TODO: update this to allow other stars? Not sure why its always v
    get_cgi_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                       phot='v', # only using V-band magnitudes as a standard
                       spt=hconf['star']['stellar_type'],
                       pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
    )


    print('Calculating initial eetc exp time')
    orig_exptime_list, orig_gain_list, orig_nframes_list, this_iter_time = get_initial_cam_params(cstrat, contrast, hconf, get_cgi_eetc, nprobepair, ndm, nlam)

    # prev lists for debugging later
    prev_exptime_list = orig_exptime_list.copy()
    prev_gain_list = orig_gain_list.copy()
    prev_nframes_list = orig_nframes_list.copy()
    iteration_durations = []
    iteration_durations.append(this_iter_time)

    # framelist
    # do last, needs peak flux
    rng = np.random.default_rng(12345)
    
    num_imager_worker = args.num_imager_worker 
    print('Using num_imager_worker = ', num_imager_worker)

    # normalisation strategy first then imager, since normalisation strategy is needed to calculate peak flux for framelist collection

    # this step is to apply probe images
    framelist = _collect_framelist(
        imager, cfg, dm1_list, dm2_list,
        exptime_list=orig_exptime_list,
        gain_list=orig_gain_list,
        nframes_list=orig_nframes_list,
        croplist=croplist,
        normalization_strategy=normalization_strategy,
        get_cgi_eetc=get_cgi_eetc,
        hconf=hconf,
        ndm=ndm,
        cstrat=cstrat,
        fracbadpix=fracbadpix,
        n_jobs=safe_cpu_count,
    )

    # drop packets for testing if requested
    if nbadpacket > 0:
        fr = rng.integers(0, len(framelist), (nbadpacket,))
        pr = rng.integers(0, nrow//nrowperpacket, (nbadpacket,))
        log.info('Dropping packets ' + str(pr) + ' of frames ' + str(fr))
        for j in range(nbadpacket):
            lr = pr[j]*nrowperpacket
            ur = (pr[j] + 1)*nrowperpacket
            framelist[fr[j]][lr:ur, :] *= np.nan
            pass
        pass


    # drop frames for testing if requested
    if nbadframe > 0:
        fr = rng.integers(0, len(framelist), (nbadframe,))
        log.info('Dropping frames ' + str(fr))
        for j in range(nbadframe):
            framelist[fr[j]] *= np.nan
            pass
        pass

    for iteration in range(1, niter+1): # var is number of next iteration

        t0 = time.time()
        if isprof:
            pr.enable()
            pass
        abs_dm1, abs_dm2, scale_factor_list, gain_list, exptime_list, \
        nframes_list, prev_c, next_c, next_time, status, other, debugging_dict = \
        howfsc_computation(framelist, dm1_list, dm2_list, cfg, jac, jtwj_map,
                           croplist, prev_exptime_list,
                           cstrat, n2clist, hconf, iteration,
                           estimator, imager, normalization_strategy, probes)
        if isprof:
            pr.disable()
            pass
        t1 = time.time()

        otherlist.append(other)
        abs_dm1list.append(abs_dm1)
        abs_dm2list.append(abs_dm2)
        framelistlist.append(framelist)
        scalelistout.append(scale_factor_list)
        camlist.append([gain_list, exptime_list, nframes_list])

        # New lists compared to original version
        measured_c.append(prev_c)
        pred_c.append(next_c)

        # Add camera parameters to debugging dictionary
        debugging_dict['cam_params']['nom'] = np.zeros((nlam, 3))
        debugging_dict['cam_params']['probing'] = np.zeros((nlam, 3))
        for j in range(nlam):
            nom_idx = j * ndm  # unprobed frame index in flat list
            probe_idx = j * ndm + 1  # first probe frame index
            debugging_dict['cam_params']['nom'][j, :] = [
                prev_gain_list[nom_idx],
                prev_exptime_list[nom_idx],
                prev_nframes_list[nom_idx],
            ]
            debugging_dict['cam_params']['probing'][j, :] = [
                prev_gain_list[probe_idx],
                prev_exptime_list[probe_idx],
                prev_nframes_list[probe_idx],
            ]

        log.info('-----------------------------------')
        log.info('Summary of iteration ' + str(iteration))
        log.info('HOWFSC computation time: ' + str(t1-t0))
        log.info('Previous contrast: ' + str(prev_c))
        log.info('Next contrast: ' + str(next_c))
        log.info('scales: ' + str(scale_factor_list))

        # Write current iterations files now
        if fileout is not None and output_every_iter:
            hdr = pyfits.Header()
            hdr['NLAM'] = len(cfg.sl_list)
            prim = pyfits.PrimaryHDU(header=hdr)
            img = pyfits.ImageHDU(framelist)
            prev = pyfits.ImageHDU(prev_exptime_list)
            hdul = pyfits.HDUList([prim, img, prev])
            hdul.writeto(fileout, overwrite=True)

            if output_model_efield and imager.backend == 'corgihowfsc':
                # if speedup == True: getting the model e-field for the central bandpass (e.g. 1b for band 1)
                # if speedup == False: getting the model e-field for all bandpasses
                if estimator.name == 'perfect':
                    perfect_efield = [otherlist[iteration-1][j]['meas_efield'] for j in range(len(cfg.sl_list))]
                else:
                    perfect_efield = imager.get_all_efields(abs_dm1=abs_dm1, abs_dm2=abs_dm2, croplist=croplist, nlam=nlam, ndm=ndm, speedup=True)

                perfect_efield_list.append(perfect_efield)
            else:
                perfect_efield_list.append(None)

            ni_score, ni_inner, ni_outer = get_ni(framelistlist[iteration-1], cfg, prev_exptime_list,
                                                  debugging_dict['peakflux'], normalization_strategy, ndm, nrow, ncol)
            ni_lists['ni_score'].append(ni_score)
            ni_lists['ni_inner'].append(ni_inner)
            ni_lists['ni_outer'].append(ni_outer)

            debugging_dict['this_iter_time'] = iteration_durations[iteration-1]
            _, _ = save_outputs_iter(iteration-1, fileout, cfg, camlist, framelistlist, otherlist, measured_c, abs_dm1list, abs_dm2list, output_every_iter, pred_c, ni_lists, perfect_efield_list[iteration-1], jac, iteration_durations=iteration_durations, debugging_dict=debugging_dict)

            # Append iteration duration for next iteration
            iteration_durations.append(debugging_dict['next_iter_dur'])
            
        
        print('-----------------------------------')
        print('Iteration: ' + str(iteration))
        print('HOWFSC computation time: ' + str(t1-t0))
        print('Previous contrast: ' + str(prev_c))
        print('Next contrast: ' + str(next_c))
        print('scales: ' + str(scale_factor_list))


        # new dm1_list, dm2_list
        dm1_list = []
        dm2_list = []
        # slight change to allow for different numbers of probe pairs when using perfect estimator to speed up iterations
        nprobepair = len(dmrel_list)
        for index in range(nlam):
            dm1_list.append(abs_dm1)  # unprobed
            for i, dmrel in enumerate(dmrel_list):
                dm1_list.append(abs_dm1 + scale_factor_list[i] * dmrel)  # positive
                dm1_list.append(abs_dm1 + scale_factor_list[i + nprobepair] * dmrel)  # negative
            for j in range(ndm):
                dm2_list.append(abs_dm2)

        # Leaving old framework here for future comparison with nulltest_gitl.py (which by design only allows 3 sets of probe commands)
        # for index in range(nlam):
        #     # DM1 same per wavelength
        #     dm1_list.append(abs_dm1)
        #     dm1_list.append(abs_dm1 + scale_factor_list[0]*dmrel_list[0])
        #     dm1_list.append(abs_dm1 + scale_factor_list[3]*dmrel_list[0])
        #     dm1_list.append(abs_dm1 + scale_factor_list[1]*dmrel_list[1])
        #     dm1_list.append(abs_dm1 + scale_factor_list[4]*dmrel_list[1])
        #     dm1_list.append(abs_dm1 + scale_factor_list[2]*dmrel_list[2])
        #     dm1_list.append(abs_dm1 + scale_factor_list[5]*dmrel_list[2])
        #     for j in range(ndm):
        #         # DM2 always same
        #         dm2_list.append(abs_dm2)
        #         pass
        #     pass

        # Skip the very last Jacobian that never gets used
        if precomp in ['precomp_jacs_always'] and iteration < niter:
            t0 = time.time()
            jac, jtwj_map, _ = howfsc_precomputation(
                cfg=cfg,
                dmset_list=[abs_dm1, abs_dm2],
                cstrat=cstrat,
                subcroplist=subcroplist,
                jacmethod='fast',
                do_n2clist=False,
                num_process=num_process,
                num_threads=num_threads,
            )
            t1 = time.time()
            log.info('Jac recalc time: ' + str(t1-t0) + ' seconds')

        # prev_[camparams]_list
        prev_exptime_list = param_order_to_list(exptime_list)
        prev_gain_list = param_order_to_list(gain_list)
        prev_nframes_list = param_order_to_list(nframes_list)

        # new framelist
        if parallel_approach == 'intuitive':
          framelist =  imager.get_images(dm1_list, dm2_list, prev_exptime_list, prev_gain_list, croplist, cstrat, hconf,
                                          normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix)
        elif parallel_approach == 'robust':
          framelist = _collect_framelist(
              imager, cfg, dm1_list, dm2_list,
              exptime_list=prev_exptime_list,
              gain_list=prev_gain_list,
              nframes_list=prev_nframes_list,
              croplist=croplist,
              normalization_strategy=normalization_strategy,
              get_cgi_eetc=get_cgi_eetc,
              hconf=hconf,
              ndm=ndm,
              cstrat=cstrat,
              fracbadpix=fracbadpix,
              n_jobs=safe_cpu_count,
          )

        # drop packets for testing if requested
        if nbadpacket > 0:
            fr = rng.integers(0, len(framelist), (nbadpacket,))
            pr = rng.integers(0, nrow//nrowperpacket, (nbadpacket,))
            log.info('Dropping packets ' + str(pr) + ' of frames ' + str(fr))
            for j in range(nbadpacket):
                lr = pr[j]*nrowperpacket
                ur = (pr[j] + 1)*nrowperpacket
                framelist[fr[j]][lr:ur, :] *= np.nan
                pass
            pass

        # drop frames for testing if requested
        if nbadframe > 0:
            fr = rng.integers(0, len(framelist), (nbadframe,))
            log.info('Dropping frames ' + str(fr))
            for j in range(nbadframe):
                framelist[fr[j]] *= np.nan
                pass
            pass

        # technically new nrow, ncol, croplist too, but these don't actually
        # change (parameters are not updated)
        pass

    if isprof:
        # cumtime
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf)
        ps.sort_stats("cumtime").print_stats("howfsc", 20)
        log.info("Profiler stats (cumtime)\n%s", buf.getvalue())

        # tottime
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf)
        ps.sort_stats("tottime").print_stats("howfsc", 20)
        log.info("Profiler stats (tottime)\n%s", buf.getvalue())

        pass

    if fileout is not None:
        hdr = pyfits.Header()
        hdr['NLAM'] = len(cfg.sl_list)
        prim = pyfits.PrimaryHDU(header=hdr)
        img = pyfits.ImageHDU(framelist)
        prev = pyfits.ImageHDU(prev_exptime_list)
        hdul = pyfits.HDUList([prim, img, prev])
        hdul.writeto(fileout, overwrite=True)

        ni_score, ni_inner, ni_outer = get_ni(framelistlist[iteration - 1], cfg, prev_exptime_list,
                                              debugging_dict['peakflux'], normalization_strategy, ndm, nrow, ncol)
        ni_lists['ni_score'].append(ni_score)
        ni_lists['ni_inner'].append(ni_inner)
        ni_lists['ni_outer'].append(ni_outer)

        save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c, abs_dm1list, abs_dm2list, output_every_iter, pred_c, ni_lists, perfect_efield_list, jac)


if __name__ == "__main__":
    # setup for cmd line args
    ap = argparse.ArgumentParser(prog='python nulltest_gitl.py', description="Run a nulling sequence, using the optical model as the data source.  Outputs will be displayed to the command line.")
    ap.add_argument('-n', '--niter', default=5, help="Number of iterations to run.  Defaults to 5.", type=int)
    ap.add_argument('--mode', default='widefov', choices=['widefov', 'narrowfov', 'spectroscopy', 'nfov_dm', 'nfov_flat'], help="coronagraph mode from test data; must be one of 'widefov', 'narrowfov', 'nfov_dm', 'nfov_flat', or 'spectroscopy'.  Defaults to 'widefov'.")
    ap.add_argument('--profile', action='store_true', help='If present, runs the Python cProfile profiler on howfsc_computation and displays the top 20 howfsc/ contributors to cumulative time')
    ap.add_argument('-p', '--fracbadpix', default=0, type=float, help="Fraction of pixels, in [0, 1], to make randomly bad (e.g. due to cosmic rays).  Defaults to 0, with no bad pixels.")
    ap.add_argument('--nbadpacket', default=0, type=int, help="Number of GITL packets (3 rows x 153 cols) irrecoverably lost due to errors, to be replaced by NaNs.  Defaults to 0.  Same number is applied to every iteration, although not in the same place.")
    ap.add_argument('--nbadframe', default=0, type=int, help="Number of entire GITL frames (153 x 153) irrecoverably lost due to errors, to be replaced by NaNs.  Defaults to 0.  Same number is applied to every iteration, although not in the same place.")
    ap.add_argument('--logfile', default=None, help="If present, absolute path to file location to log to.")
    ap.add_argument('--precomp', default='load_all', choices=['load_all', 'precomp_all_once', 'precomp_jacs_once', 'precomp_jacs_always'], help="Specifies Jacobian precomputation behavior.  'load_all' will load everything from files at the start and leave them fixed.  'precomp_all_once' will compute a Jacobian, JTWJs, and n2clist once at start.  'precomp_jacs_once' will compute a Jacobian and JTWJs once at start, and load n2clist from file.  'precomp_jacs_always' will compute a Jacobian and JTWJs at start and at every iteration; n2clist will be loaded from file.")

    ap.add_argument(
        '--num_process', default=None,
        help="number of processes, defaults to None, indicating no multiprocessing. If 0, then howfsc_precomputation() defaults to the available number of cores", type=int
    )
    ap.add_argument(
        '--num_threads', default=None,
        help="set mkl_num_threads to use for parallel processes for calcjacs(). If None (default), then do nothing (number of threads might also be set externally through environment variable 'MKL_NUM_THREADS' or 'HOWFS_CALCJAC_NUM_THREADS'",
        type=int
    )

    ap.add_argument('--fileout', default=None, help="If present, absolute path to file location of output .fits (including \'.fits\' at the end) file containing framelist and prev_exptime_list, as well as nlam stored as a header.")
    ap.add_argument('--stellarvmag', default=None, type=float, help='If present, magnitude of the reference star desired will be updated in the hconf file (for parameter stellar_vmag in hconf file).')
    ap.add_argument('--stellartype', default=None, help='If present, type of the reference star desired will be updated in the hconf file (for parameter stellar_type in hconf file).')
    ap.add_argument('--stellarvmagtarget', default=None, type=float, help='If present, magnitude of the target star desired will be updated in the hconf file (for parameter stellar_vmag_target in hconf file).')
    ap.add_argument('--stellartypetarget', default=None, help='If present, type of the target star desired will be updated in the hconf file (for parameter stellar_type_target in hconf file).')

    ap.add_argument('-j', '--jacpath', default=defjacpath, help="absolute path to read Jacobian files from", type=str)
    args = ap.parse_args()

    # User params
    niter = args.niter
    mode = args.mode
    isprof = args.profile
    logfile = args.logfile
    fracbadpix = args.fracbadpix
    nbadpacket = args.nbadpacket
    nbadframe = args.nbadframe
    fileout = args.fileout
    stellar_vmag = args.stellarvmag
    stellar_type = args.stellartype
    stellar_vmag_target = args.stellarvmagtarget
    stellar_type_target = args.stellartypetarget
    jacpath = args.jacpath

    precomp = args.precomp
    num_process = args.num_process
    num_threads = args.num_threads

    nulling_gitl(niter, mode, isprof, logfile, fracbadpix, nbadpacket,
                      nbadframe, fileout, stellar_vmag, stellar_type,
                      stellar_vmag_target, stellar_type_target, jacpath,
                      precomp, num_process, num_threads)
