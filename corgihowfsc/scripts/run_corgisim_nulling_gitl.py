
import time
import os
import argparse
import cProfile
import pstats
import logging
from datetime import datetime
import numpy as np
import astropy.io.fits as pyfits

import corgihowfsc
import eetc
from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml

import roman_preflight_proper
### Then, run the following command to copy the default prescription file
roman_preflight_proper.copy_here()

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.PerfectEstimator import PerfectEstimator
from corgihowfsc.sensing.DefaultProbes import DefaultProbes
from corgihowfsc.utils.contrast_nomalization import CorgiNormalization, EETCNormalization
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')  # User should set to somewhere outside the repo

# Note: MUST DEFINE JACPATH FOR CORGI GITL FRAMES
precomp = 'precomp_jacs_always' #'load_all' if defjacpath is not None else 'precomp_all_once'
current_datetime = datetime.now()
folder_name = 'gitl_simulation_' + current_datetime.strftime("%Y-%m-%d_%H%M%S")
fits_name = 'final_frames.fits'
fileout_path = os.path.join(os.path.dirname(os.path.dirname(corgihowfsc.__file__)), 'data', folder_name, fits_name)
dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']

def main(): 
    args = get_args(mode='nfov_band1',
                    precomp=precomp,
                    num_process=2,
                    num_threads=1,
                    fileout=fileout_path,
                    jacpath=defjacpath,
                    dmstartmap_filenames=dmstartmap_filenames)

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

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)

    # cfg
    cfg = CoronagraphMode(cfgfile)

    # hconffile
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    # Define control and estimator strategy
    cstrat = ControlStrategy(cstratfile)
    estimator = DefaultEstimator()

    # Initialize default probes class
    probes = DefaultProbes('default')

    # Image cropping parameters:
    crop_params = {}
    crop_params['nrow'] = 153
    crop_params['ncol'] = 153
    crop_params['lrow'] = 0
    crop_params['lcol'] = 0

    # Define imager and normalization (counts->contrast) strategy
    corgi_overrides = {}
    corgi_overrides['output_dim'] = crop_params['nrow']
    imager = GitlImage(
        cfg=cfg,         # Your CoronagraphMode object
        cstrat=cstrat,   # Your ControlStrategy object
        hconf=hconf,      # Your host config with stellar properties
        backend='corgihowfsc',
        cor=mode,
        corgi_overrides=corgi_overrides
    )

    normalization_strategy = CorgiNormalization(cfg, cstrat, hconf, cor=args.mode, corgi_overrides=corgi_overrides, separation_lamD=7, exptime_norm=0.1)
    # normalization_strategy = EETCNormalization()

    nulling_gitl(cstrat, estimator, probes, normalization_strategy, imager, cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles, crop_params, dmstartmaps)

if __name__ == '__main__':    
    main()
