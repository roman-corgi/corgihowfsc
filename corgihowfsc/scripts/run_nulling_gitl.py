
import time
import os
import argparse
import cProfile
import pstats
import logging

import numpy as np
import astropy.io.fits as pyfits

import howfsc
import eetc
from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode

from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.DefaultProbes import DefaultProbes
from corgihowfsc.utils.contrast_nomalization import EETCNormalization
# from corgihowfsc.utils.Imager import Imager
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')

defjacpath = r'C:\Users\sredmond\Documents\github_repos\roman-corgi-repos\cgi-howfsc'
args = get_args(jacpath=defjacpath)

# Initialize variables etc

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

modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles = load_files(args, howfscpath)

# Initialize default probes class
probing = DefaultProbes('default')

# # Get DM lists
# probefiles = {}
# probefiles[0] = probe0file
# probefiles[2] = probe1file
# probefiles[1] = probe2file
# dm1_list, dm2_list = probing.get_dm_probes(cfg, probefiles,
#                       scalelist=[0.3, 0.3, 0.3, -0.3, -0.3, -0.3])
#
# # Get probe amplitude lists (this step is slow)
# plist, other = probing.get_probe_ap(cfg, dm1_list, dm2_list)

# cfg
cfg = CoronagraphMode(cfgfile)

cstrat = ControlStrategy(cstratfile)
estrat = DefaultEstimator()

# imager = corgisimImager()
imager = GitlImage("cgi-howfsc", cfg=cfg)
normstrat = EETCNormalization()

nulling_gitl(cstrat, estrat, probing, normstrat, imager, cfg, args, modelpath, jacfile, probefiles, hconffile, n2clistfiles)

