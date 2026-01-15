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
import logging

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

from howfsc.gitl import howfsc_computation
from howfsc.precomp import howfsc_precomputation

# from howfsc.scripts.gitlframes import sim_gitlframe
from howfsc.scripts.cgisim_gitl_frames import gen_cgisim_excam_frame

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))

if __name__ == "__main__":
    defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')

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

    otherlist = []
    abs_dm1list = []
    abs_dm2list = []
    framelistlist = []
    scalelistout = []
    camlist = []

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

    exptime = 10 # FIXME this should be derived from contrast eventually
    contrast = 1e-5 # "starting" value to bootstrap getting we0

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
    elif mode == 'widefov':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'widefov')
        cfgfile = os.path.join(modelpath, 'widefov.yaml')
        jacfile = os.path.join(jacpath, 'widefov_jac_full.fits')
        cstratfile = os.path.join(modelpath, 'cstrat_widefov.yaml')
        probe0file = os.path.join(modelpath,
                                  'widefov_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(modelpath,
                                  'widefov_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(modelpath,
                                  'widefov_dmrel_1.0e-05_sinud.fits')
        hconffile = os.path.join(modelpath, 'hconf_widefov.yaml')
        n2clistfiles = [
            os.path.join(modelpath, 'widefov_n2c_idx0.fits'),
            os.path.join(modelpath, 'widefov_n2c_idx1.fits'),
            os.path.join(modelpath, 'widefov_n2c_idx2.fits'),
        ]
        pass
    elif mode == 'spectroscopy':
        modelpath = os.path.join(howfscpath, 'model', 'testdata',
                                 'spectroscopy')
        cfgfile = os.path.join(modelpath, 'spectroscopy.yaml')
        jacfile = os.path.join(jacpath, 'spectroscopy_jac_full.fits')
        cstratfile = os.path.join(modelpath, 'cstrat_spectroscopy.yaml')
        probe0file = os.path.join(modelpath,
                                  'spectroscopy_dmrel_1.0e-05_cos.fits')
        probe1file = os.path.join(modelpath,
                                  'spectroscopy_dmrel_1.0e-05_sinlr.fits')
        probe2file = os.path.join(modelpath,
                                  'spectroscopy_dmrel_1.0e-05_sinud.fits')
        hconffile = os.path.join(modelpath, 'hconf_spectroscopy.yaml')
        n2clistfiles = [
            os.path.join(modelpath, 'spectroscopy_n2c_idx0.fits'),
            os.path.join(modelpath, 'spectroscopy_n2c_idx1.fits'),
            os.path.join(modelpath, 'spectroscopy_n2c_idx2.fits'),
            os.path.join(modelpath, 'spectroscopy_n2c_idx3.fits'),
            os.path.join(modelpath, 'spectroscopy_n2c_idx4.fits'),
        ]
        pass
    else:
        # should not reach here; argparse should catch this
        raise ValueError('Invalid coronagraph mode type')


    # cfg
    cfg = CoronagraphMode(cfgfile)

    # dm1_list, dm2
    dmrel_list = [pyfits.getdata(probe0file),
                  pyfits.getdata(probe1file),
                  pyfits.getdata(probe2file),
                  ] # these are 1e-5 probe relative DM settings
    # relative to 1e-5 for these, sqrt (0.1 = 1e-7)
    scalelist = [0.3, 0.3, 0.3, -0.3, -0.3, -0.3]

    nlam = len(cfg.sl_list)
    ndm = 2*len(dmrel_list) + 1

    dm10 = cfg.initmaps[0]
    dm20 = cfg.initmaps[1]
    dm1_list = []
    dm2_list = []
    for index in range(nlam):
        # DM1 same per wavelength
        dm1_list.append(dm10)
        dm1_list.append(dm10 + scalelist[0]*dmrel_list[0])
        dm1_list.append(dm10 + scalelist[3]*dmrel_list[0])
        dm1_list.append(dm10 + scalelist[1]*dmrel_list[1])
        dm1_list.append(dm10 + scalelist[4]*dmrel_list[1])
        dm1_list.append(dm10 + scalelist[2]*dmrel_list[2])
        dm1_list.append(dm10 + scalelist[5]*dmrel_list[2])
        for j in range(ndm):
            # DM2 always same
            dm2_list.append(dm20)
            pass
        pass

    # cstratfile
    cstrat = ControlStrategy(cstratfile)

    # nrow, ncol, croplist
    nrow = 153
    ncol = 153
    lrow = 436
    lcol = 436
    croplist = [(lrow, lcol, nrow, ncol)]*(nlam*ndm)
    subcroplist = [(lrow, lcol, nrow, ncol)]*(nlam)
    nrowperpacket = 3 # only used by packet-drop testing

    # prev_exptime_list
    prev_exptime_list = [exptime]*(nlam*ndm)

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
        pass
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

    # hconffile
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    if stellar_vmag is not None:
        hconf['star']['stellar_vmag'] = stellar_vmag
    if stellar_type is not None:
        hconf['star']['stellar_type'] = stellar_type
    if stellar_vmag_target is not None:
        hconf['star']['stellar_vmag_target'] = stellar_vmag_target
    if stellar_type_target is not None:
        hconf['star']['stellar_type_target'] = stellar_type_target

    cgi_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                       phot='v', # only using V-band magnitudes as a standard
                       spt=hconf['star']['stellar_type'],
                       pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
    )

    # framelist
    # do last, needs peak flux
    rng = np.random.default_rng(12345)
    framelist = []
    for indj, sl in enumerate(cfg.sl_list):
        crop = croplist[indj]
        
        _, peakflux = cgi_eetc.calc_flux_rate(
            sequence_name=hconf['hardware']['sequence_list'][indj],
        )
        
        for indk in range(ndm):
#             dmlist = [dm1_list[indj*ndm + indk],
#                       dm2_list[indj*ndm + indk]]
            dm1v = dm1_list[indj*ndm + indk]
            dm2v = dm2_list[indj*ndm + indk]
            f = gen_cgisim_excam_frame(exptime, gain, dm1v, dm2v, cor, bandpass, crop, 
                fixedbp=cstrat.fixedbp)
#             f = sim_gitlframe(cfg,
#                               dmlist,
#                               cstrat.fixedbp,
#                               peakflux,
#                               exptime,
#                               crop,
#                               indj)
            # bpmeas = rng.random(f.shape) > (1 - args.fracbadpix)
            # f[bpmeas] = np.nan
            framelist.append(f)
            pass
        pass

#     # drop packets for testing if requested
#     if nbadpacket > 0:
#         fr = rng.integers(0, len(framelist), (nbadpacket,))
#         pr = rng.integers(0, nrow//nrowperpacket, (nbadpacket,))
#         log.info('Dropping packets ' + str(pr) + ' of frames ' + str(fr))
#         for j in range(nbadpacket):
#             lr = pr[j]*nrowperpacket
#             ur = (pr[j] + 1)*nrowperpacket
#             framelist[fr[j]][lr:ur, :] *= np.nan
#             pass
#         pass


#     # drop frames for testing if requested
#     if nbadframe > 0:
#         fr = rng.integers(0, len(framelist), (nbadframe,))
#         log.info('Dropping frames ' + str(fr))
#         for j in range(nbadframe):
#             framelist[fr[j]] *= np.nan
#             pass
#         pass

    for iteration in range(1, niter+1): # var is number of next iteration

        t0 = time.time()
        if isprof:
            pr.enable()
            pass
        abs_dm1, abs_dm2, scale_factor_list, gain_list, exptime_list, \
        nframes_list, prev_c, next_c, next_time, status, other = \
        howfsc_computation(framelist, dm1_list, dm2_list, cfg, jac, jtwj_map,
                           croplist, prev_exptime_list,
                           cstrat, n2clist, hconf, iteration)
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

        print('-----------------------------------')
        print('Iteration: ' + str(iteration))
        print('HOWFSC computation time: ' + str(t1-t0))
        print('Previous contrast: ' + str(prev_c))
        print('Next contrast: ' + str(next_c))
        print('scales: ' + str(scale_factor_list))


        # new dm1_list, dm2_list
        dm1_list = []
        dm2_list = []
        for index in range(nlam):
            # DM1 same per wavelength
            dm1_list.append(abs_dm1)
            dm1_list.append(abs_dm1 + scale_factor_list[0]*dmrel_list[0])
            dm1_list.append(abs_dm1 + scale_factor_list[3]*dmrel_list[0])
            dm1_list.append(abs_dm1 + scale_factor_list[1]*dmrel_list[1])
            dm1_list.append(abs_dm1 + scale_factor_list[4]*dmrel_list[1])
            dm1_list.append(abs_dm1 + scale_factor_list[2]*dmrel_list[2])
            dm1_list.append(abs_dm1 + scale_factor_list[5]*dmrel_list[2])
            for j in range(ndm):
                # DM2 always same
                dm2_list.append(abs_dm2)
                pass
            pass

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
            print('Jac recalc time: ' + str(t1-t0) + ' seconds')

        # prev_exptime_list
        prev_exptime_list = param_order_to_list(exptime_list)

        # new framelist
        framelist = []
        for indj, sl in enumerate(cfg.sl_list):
            crop = croplist[indj]
            _, peakflux = cgi_eetc.calc_flux_rate(
                sequence_name=hconf['hardware']['sequence_list'][indj],
            )
            for indk in range(ndm):
                dmlist = [dm1_list[indj*ndm + indk],
                          dm2_list[indj*ndm + indk]]
                f = sim_gitlframe(cfg,
                                  dmlist,
                                  cstrat.fixedbp,
                                  peakflux,
                                  prev_exptime_list[indj*ndm + indk],
                                  crop,
                                  indj,
                                  cleanrow=hconf['excam']['cleanrow'],
                                  cleancol=hconf['excam']['cleancol'])
                framelist.append(f)
                pass
            pass

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
        ps = pstats.Stats(pr)
        ps.sort_stats('cumtime').print_stats('howfsc', 20)

        print() # line separate tottime
        ps.sort_stats('tottime').print_stats('howfsc', 20)

        pass

    if fileout is not None:
        hdr = pyfits.Header()
        hdr['NLAM'] = len(cfg.sl_list)
        prim = pyfits.PrimaryHDU(header=hdr)
        img = pyfits.ImageHDU(framelist)
        prev = pyfits.ImageHDU(prev_exptime_list)
        hdul = pyfits.HDUList([prim, img, prev])
        hdul.writeto(fileout, overwrite=True)
