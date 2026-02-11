import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')

import eetc
from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.PerfectEstimator import PerfectEstimator
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_nomalization import CorgiNormalization, EETCNormalization
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.make_output_file_structure import make_output_file_structure

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')  # User should set to somewhere outside the repo

base_path = Path.home()  # this is the proposed default but can be changed
base_corgiloop_path = 'corgiloop_data'
final_filename = 'final_frames.fits'

loop_framework = 'cgi-howfsc' # do not modify
backend_type = 'cgi-howfsc' # Note: the framework cgi-howfsc can only work with the compact model

precomp = 'precomp_jacs_always' #'load_all' if defjacpath is not None else 'precomp_all_once'
dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']

fileout_path = make_output_file_structure(loop_framework, backend_type, base_path, base_corgiloop_path, final_filename)

def main(): 

    args = get_args(
        niter=5,
        mode='nfov_band1',
        dark_hole='360deg',
        probe_shape='default',
        precomp=precomp,
        num_process=0,
        num_threads=1,
        fileout=fileout_path,
        jacpath=defjacpath,
        dmstartmap_filenames=dmstartmap_filenames,
    )

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

    # Control strategy and estimator
    cstrat = ControlStrategy(cstratfile)
    estimator = DefaultEstimator() # PerfectEstimator() will use the exact efield to make EFC, DefaultEstimator will use PWP.
    probes = ProbesShapes(args.probe_shape)

    # Image cropping parameters:
    crop_params = {}
    crop_params['nrow'] = 153
    crop_params['ncol'] = 153
    crop_params['lrow'] = 436
    crop_params['lcol'] = 436

    # Define imager and normalization (counts->contrast) strategy
    imager = GitlImage(
        cfg=cfg,         # Your CoronagraphMode object
        cstrat=cstrat,   # Your ControlStrategy object
        hconf=hconf,      # Your host config with stellar properties
        backend='cgi-howfsc',
        cor=mode
    )
    normalization_strategy = EETCNormalization()
    nulling_gitl(cstrat, estimator, probes, normalization_strategy, imager, cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles, crop_params, dmstartmaps)

if __name__ == '__main__':    
    main()