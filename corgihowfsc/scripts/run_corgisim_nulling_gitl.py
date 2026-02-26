import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')

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
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_nomalization import CorgiNormalization, EETCNormalization
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.output_management import make_output_file_structure

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'temp')  # User should set to somewhere outside the repo
precomp = 'precomp_jacs_always' #'load_all' if defjacpath is not None else 'precomp_all_once'

base_path = Path.home()  # this is the proposed default but can be changed
base_corgiloop_path = 'corgiloop_data'
final_filename = 'final_frames.fits'

loop_framework = 'corgi-howfsc' # do not modify
backend_type = 'cgi-howfsc'  # 'corgihowfsc' for the corgisim model, otherwise for the compact model use: 'cgi-howfsc'

dmstartmap_filenames = ['iter_080_dm1.fits', 'iter_080_dm2.fits']

fileout_path = make_output_file_structure(loop_framework, backend_type, base_path, base_corgiloop_path, final_filename)

def main():

    args = get_args(
        niter=5,
        mode='nfov_band1',
        dark_hole='360deg',
        probe_shape='default',
        precomp=precomp,
        num_process=2,
        num_threads=1,
        fileout=fileout_path,
        jacpath=defjacpath,
        dmstartmap_filenames=dmstartmap_filenames,
    )

    # User params
    mode = args.mode

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)

    # cfg
    cfg = CoronagraphMode(cfgfile)

    # hconffile
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    # Define control and estimator strategy
    cstrat = ControlStrategy(cstratfile)
    estimator = DefaultEstimator()

    # Initialize default probes class
    probes = ProbesShapes(args.probe_shape)

    # Image cropping parameters:
    crop_params = {}
    crop_params['nrow'] = 153 # FIXED VALUE; do not change this
    crop_params['ncol'] = 153 # FIXED VALUE; do not change this

    # Define imager and normalization (counts->contrast) strategy
    corgi_overrides = {}
    corgi_overrides['output_dim'] = crop_params['nrow']
    corgi_overrides['is_noise_free'] = False
    corgi_overrides['oversampling_factor'] = 2    
    
    imager = GitlImage(
        cfg=cfg,         # Your CoronagraphMode object
        cstrat=cstrat,   # Your ControlStrategy object
        hconf=hconf,      # Your host config with stellar properties
        backend=backend_type,
        cor=mode,
        corgi_overrides=corgi_overrides
    )

    if backend_type == 'cgi-howfsc':
        crop_params['lrow'] = 436
        crop_params['lcol'] = 436

        normalization_strategy = EETCNormalization()
        
    elif backend_type == 'corgihowfsc':
        crop_params['lrow'] = 0
        crop_params['lcol'] = 0  
        normalization_strategy = CorgiNormalization(cfg,
                                                cstrat,
                                                hconf,
                                                cor=args.mode,
                                                corgi_overrides=corgi_overrides,
                                                separation_lamD=7,
                                                exptime_norm=0.01)

    metadata = {
        "inputs": {
            "modelpath": str(modelpath),
            "cfgfile": str(cfgfile),
            "hconffile": str(hconffile),
            "cstratfile": str(cstratfile),
            "jacfile": str(jacfile),
            "probefiles": {str(k): str(v) for k, v in probefiles.items()} 
            if isinstance(probefiles, dict) else probefiles,
            "n2clistfiles": [str(p) for p in (n2clistfiles or [])],
        },
        "hconf": hconf,  # already YAML-safe
        "objects": {
            "cfg_class": type(cfg).__name__,
            "cstrat_class": type(cstrat).__name__,
            "estimator_class": type(estimator).__name__,
            "probes_class": type(probes).__name__,
            "imager_class": type(imager).__name__,
        },
        "crop_params": crop_params,
        "corgi_overrides": corgi_overrides,
    }

    nulling_gitl(cstrat, estimator, probes, normalization_strategy, imager, cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles, crop_params, dmstartmaps, metadata)


if __name__ == '__main__':    
    main()
