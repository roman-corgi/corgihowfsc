import os
from pathlib import Path
import matplotlib
import argparse

matplotlib.use('TkAgg')

import eetc
from howfsc.control.cs import ControlStrategy
from howfsc.model.mode import CoronagraphMode
from howfsc.util.loadyaml import loadyaml

import roman_preflight_proper

### Then, run the following command to copy the default prescription file
roman_preflight_proper.copy_here()

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files, get_cpu_allocation
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.PerfectEstimator import PerfectEstimator
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_nomalization import CorgiNormalization, EETCNormalization, CorgiNormalizationOnAxis
from corgihowfsc.gitl.nulling_gitl import nulling_gitl
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.output_management import make_output_file_structure

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))


def main():

    # Set the path to the default parameter file relative to this script
    default_param_file = os.path.join(os.path.dirname(__file__), 'default_param.yml')

    # Create the argument parser and add the --param_file argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--param_file',
        type=str,
        default=default_param_file,
        help='Path to parameter YAML file'
    )

    cmd_args = parser.parse_args()
    param_file = os.path.abspath(os.path.expanduser(cmd_args.param_file))

    if not os.path.isfile(param_file):
        raise FileNotFoundError(f'Parameter file not found: {param_file}')

    params = loadyaml(param_file)

    # Shared config
    active_model = params['active_model']

    runtime = params['runtime']
    sim_settings = params['sim_settings']
    paths = params['paths']
    crop_cfg = params['crop']
    model_cfg = params['models'][active_model]

    # Simulation settings
    loop_framework = sim_settings['loop_framework']
    precomp = sim_settings['precomp']
    output_every_iter = sim_settings['output_every_iter']
    niter = sim_settings['niter']
    mode = sim_settings['mode']
    dark_hole = sim_settings['dark_hole']
    probe_shape = sim_settings['probe_shape']

    # Backend specific settings - which imager model to use, normalisation strategy, and which dmstartmaps to use for the first iteration (if any)
    backend_type = model_cfg['backend_type']
    normalization_type = model_cfg['normalization_type']
    dmstartmap_filenames = model_cfg['dmstartmap_filenames']

    # Paths
    base_path = Path(paths['base_path']).expanduser()
    base_corgiloop_path = paths['base_corgiloop_path']
    final_filename = paths['final_filename']
    folder_tag = paths['folder_tag']

    defjacpath_cfg = paths['defjacpath']
    if os.path.isabs(defjacpath_cfg):
        defjacpath = defjacpath_cfg
    else:
        defjacpath = os.path.join(os.path.dirname(howfscpath), defjacpath_cfg)

    # runtime settings
    num_proper_process = runtime['num_proper_process']
    num_jac_process = runtime['num_jac_process']
    num_imager_worker = runtime['num_imager_worker']

    print(
        backend_type,
        'nulling Gitl simulation starting with mode = {}, dark hole = {}, probe shape = {}'.format(
            mode, dark_hole, probe_shape
        )
    )

    # Output path
    fileout_path = make_output_file_structure(
        loop_framework,
        backend_type,
        base_path,
        base_corgiloop_path,
        final_filename,
        tag=folder_tag
    )

    # Validate CPU allocation
    num_jac_process, num_imager_worker, num_proper_process = get_cpu_allocation(
        num_jac_process,
        num_imager_worker,
        num_proper_process
    )

    args = get_args(
        niter=niter,
        mode=mode,
        dark_hole=dark_hole,
        probe_shape=probe_shape,
        precomp=precomp,
        num_process=num_jac_process,
        num_threads=1, # Do not change this number
        fileout=fileout_path,
        jacpath=defjacpath,
        dmstartmap_filenames=dmstartmap_filenames,
        logfile=os.path.join(os.path.dirname(fileout_path), 'gitl.log')
    )

    args.starting_contrast = float(model_cfg['starting_contrast'])
    args.num_imager_worker = num_imager_worker
    args.num_proper_process = num_proper_process

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args,
                                                                                                           howfscpath)

    cfg = CoronagraphMode(cfgfile)
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    # Define control and estimator strategy
    cstrat = ControlStrategy(cstratfile)

    # Initialize default probes class
    probes = ProbesShapes(args.probe_shape)

    # Crop parameters
    crop_params = {
        'nrow': crop_cfg['nrow'],
        'ncol': crop_cfg['ncol'],
        'lrow': model_cfg['lrow'],
        'lcol': model_cfg['lcol'],
    }

    # Corgi overrides
    corgi_overrides = model_cfg.get('corgi_overrides', {}).copy()   
    corgi_overrides['output_dim'] = crop_params['nrow']

    if num_proper_process is not None:
        corgi_overrides['NCPUS'] = num_proper_process

    if num_proper_process is not None:
        corgi_overrides['NCPUS'] = num_proper_process
        
    imager = GitlImage(
        cfg=cfg,  # Your CoronagraphMode object
        cstrat=cstrat,  # Your ControlStrategy object
        hconf=hconf,  # Your host config with stellar properties
        backend=backend_type,
        cor=mode,
        corgi_overrides=corgi_overrides
    )

    # Estimator selection
    if model_cfg['estimator'] == 'perfect':
        estimator = PerfectEstimator()
    elif model_cfg['estimator'] == 'default':
        estimator = DefaultEstimator()
    else:
        raise ValueError(f"Invalid estimator choice: {model_cfg['estimator']}. Choose 'perfect' or 'default'.")

    # Normalization
    if normalization_type == 'eetc':
        normalization_strategy = EETCNormalization(backend_type, corgi_overrides)

    elif normalization_type == 'corgisim-off-axis' and backend_type == 'corgihowfsc':
        normalization_strategy = CorgiNormalization(cfg,
                                                  cstrat,
                                                  hconf,
                                                  cor=args.mode,
                                                  corgi_overrides=corgi_overrides,
                                                  separation_lamD=7,
                                                  exptime_norm=0.01)
   
    elif normalization_type == 'corgisim-on-axis' and backend_type == 'corgihowfsc':
        normalization_strategy = CorgiNormalizationOnAxis(cfg,
                                                        cstrat,
                                                        hconf,
                                                        cor=args.mode,
                                                        corgi_overrides=corgi_overrides,
                                                        exptime_norm=0.01)
    else:
      raise ValueError('Invalid normalization type or backend-normalization combo.')


    metadata = {
        # --- user run settings ---
        "active_model": active_model,
        "backend_type": backend_type,
        "normalization_type": normalization_type,
        "niter": args.niter,
        "mode": args.mode,
        "dark_hole": args.dark_hole,
        "probe_shape": args.probe_shape,
        "precomp": args.precomp,
        # --- runtime ---
        "num_process": args.num_process,
        "num_threads": args.num_threads,
        "num_imager_worker": args.num_imager_worker,
        "num_proper_process": args.num_proper_process,
        # --- crop & overrides ---
        "crop_params": crop_params,
        "corgi_overrides": corgi_overrides,
        # --- resolved file paths ---
        "fileout": str(args.fileout),
        "jacpath": str(args.jacpath),
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
        # --- full hconf dump ---
        "hconf": hconf,
        # --- objects used ---
        "objects": {
            "cfg_class": type(cfg).__name__,
            "cstrat_class": type(cstrat).__name__,
            "estimator_class": type(estimator).__name__,
            "probes_class": type(probes).__name__,
            "imager_class": type(imager).__name__,
        },
    }

    nulling_gitl(cstrat,
                 estimator,
                 probes,
                 normalization_strategy,
                 imager,
                 cfg, args, hconf, modelpath, jacfile, probefiles, n2clistfiles, crop_params, dmstartmaps,
                 metadata, output_every_iter)


if __name__ == '__main__':
    main()