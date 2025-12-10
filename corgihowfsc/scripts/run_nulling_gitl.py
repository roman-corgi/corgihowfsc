
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


import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.DefaultProbes import DefaultProbes
from corgihowfsc.utils.contrast_nomalization import EETCNormalization
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')

precomp= 'load_all' if defjacpath is not None else 'precomp_all_once'

current_datetime = datetime.now()
folder_name = 'gitl_simulation_' + current_datetime.strftime("%Y-%m-%d_%H%M%S")
fits_name = 'final_frames.fits'
fileout_path = os.path.join(os.path.dirname(os.path.dirname(corgihowfsc.__file__)), 'data', folder_name, fits_name)

args = get_args(mode='nfov_band1',
                dark_hole='360deg',
                probe_shape='default',
                precomp=precomp,
                num_process=0,
                num_threads=1,
                fileout=fileout_path,
                jacpath=defjacpath)

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

modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles = load_files(args, howfscpath)

# cfg
cfg = CoronagraphMode(cfgfile)

# hconffile
hconf = loadyaml(hconffile, custom_exception=TypeError)

# Define control and estimator strategy
cstrat = ControlStrategy(cstratfile)
estimator = DefaultEstimator()

# Initialize default probes class
probes = DefaultProbes(args.probe_shape)

# Define imager and normalization (counts->contrast) strategy
imager = GitlImage(
    cfg=cfg,         # Your CoronagraphMode object
    cstrat=cstrat,   # Your ControlStrategy object
    hconf=hconf,      # Your host config with stellar properties
    backend='cgi-howfsc',
    cor=mode
)
normalization_strategy = EETCNormalization()

nulling_gitl(cstrat, estimator, probes, normalization_strategy, imager, cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles)

