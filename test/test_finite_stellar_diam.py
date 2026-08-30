from pathlib import Path

import time
import os
import argparse
import cProfile
import pstats
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from multiprocessing import Process,Manager,freeze_support,cpu_count
from itertools import repeat

import numpy as np
import astropy.io.fits as pyfits
import corgihowfsc

from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.sensing.DefaultEstimator import DefaultEstimator
from corgihowfsc.sensing.GettingProbes import ProbesShapes
from corgihowfsc.utils.contrast_nomalization import CorgiNormalization, EETCNormalization
from corgihowfsc.utils.corgisim_gitl_frames import GitlImage
from corgihowfsc.utils.output_management import make_output_file_structure


import eetc
from eetc.cgi_eetc import CGIEETC

import howfsc
from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjtwj import JTWJMap

from howfsc.model.mode import CoronagraphMode

from howfsc.util.loadyaml import loadyaml
from howfsc.util.gitl_tools import param_order_to_list, as_f32_normal


from howfsc.gitl import howfsc_computation
from howfsc.precomp import howfsc_precomputation

from howfsc.scripts.gitlframes import sim_gitlframe

import roman_preflight_proper
### Then, run the following command to copy the default prescription file 
roman_preflight_proper.copy_here()

import corgisim
from corgisim import jitter
#------------------------------------------------------------------------------

def simulate_test_image(use_finite_stellar_diam):
    '''
    This minimal test verifies that nothing has broken the use of the finite
    stellar diameter model in corgisim.
    '''
    
    eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
    howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))


    defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')
    # defjacpath = os.path.join(os.path.dirname(os.path.abspath(corgihowfsc.__file__)), '..', 'data')
    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))

    precomp = 'precomp_jacs_always'  # 'load_all' if defjacpath is not None else 'precomp_all_once'
 
    base_path = Path.home()  # this is the proposed default but can be changed 
    base_corgiloop_path = 'corgiloop_data'
    final_filename = 'final_frames.fits'

    loop_framework = 'corgi-howfsc'  # do not modify
    #backend_type = 'cgi-howfsc'  # 'corgihowfsc' for the corgisim model, otherwise for the compact model use: 'cgi-howfsc'

    dmstartmap_filenames = ['iter_080_dm1.fits',
                        'iter_080_dm2.fits']  # For nfov_band1 only. ['iter_061_dm1.fits', 'iter_061_dm2.fits'] for wfov_band4

    output_every_iter = True  # Set to True to save frames at every iteration in real time, False to save all data after the simulation is complete. The file structure will be the same in both cases.

    backend_type = 'corgihowfsc'  # 'corgihowfsc' for the corgisim model, otherwise for the compact model use: 'cgi-howfsc'
                                  # DO NOT CHANGE FOR THIS TEST

    mode = 'nfov_band1'
    dark_hole = '360deg'
    probe_shape = 'default'

    # Make output path
    # Note get_args() needs a fileout_path so if it is desired to move make_output_file_structure() after get_args()
    # make sure to update args.fileout with final fileout_path
    folder_tag = None  # optional additional descriptor for output folder
    fileout_path = make_output_file_structure(loop_framework, backend_type, base_path, base_corgiloop_path,
                                          final_filename, tag=folder_tag)

    args = get_args(
            niter=3,
            mode=mode,
            dark_hole=dark_hole,
            probe_shape=probe_shape,
            precomp=precomp,
            num_process=2,
            num_threads=1,
            fileout=fileout_path,
            jacpath=defjacpath,
            dmstartmap_filenames=dmstartmap_filenames,
            logfile=os.path.join(os.path.dirname(fileout_path), 'gitl.log')
        )

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

    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)



    # cfg
    cfg = CoronagraphMode(cfgfile)

    # Initialize default probes class
    probes = ProbesShapes('default')
    dm1_list, dm2_list, dmrel_list, dm10, dm20 = probes.get_dm_probes(cfg, probefiles, dmstartmaps, scalelist=[0.3, 0.3, 0.3, -0.3, -0.3, -0.3])
    hconf = loadyaml(hconffile, custom_exception=TypeError)

    cstrat = ControlStrategy(cstratfile)
    estimator = DefaultEstimator()


    # Image cropping parameters:
    crop_params = {}
    crop_params['nrow'] = 153  # FIXED VALUE; do not change this
    crop_params['ncol'] = 153  # FIXED VALUE; do not change this



    corgi_overrides={}
    corgi_overrides['output_dim'] = crop_params['nrow']
    corgi_overrides['is_noise_free'] = True
    if use_finite_stellar_diam == True:
        stellar_diam_and_jitter_keywords = jitter.load_predefined_jitter_and_stellar_diam_params(mintest=True,stellar_diam_mas=hconf['star']['stellar_diam_mas'])
        corgi_overrides['stellar_diam_and_jitter_keywords']=stellar_diam_and_jitter_keywords


    if backend_type == 'cgi-howfsc':
        crop_params['lrow'] = 436
        crop_params['lcol'] = 436

    elif backend_type == 'corgihowfsc':
        crop_params['lrow'] = 0
        crop_params['lcol'] = 0

    ###############################################################################
    # THIS IS WHERE YOU CAN CHAGE SERIAL->PARALLEL
    imager = GitlImage(
        cfg=cfg,         # Your CoronagraphMode object
        cstrat=cstrat,   # Your ControlStrategy object  
        hconf=hconf,      # Your host config with stellar properties
        backend=backend_type, 
        cor=mode,
        corgi_overrides=corgi_overrides
    )

    crop_corgi = (crop_params['lrow'], crop_params['lcol'], crop_params['nrow'], crop_params['ncol'])


    normalization_strategy = EETCNormalization(backend_type,corgi_overrides)

    get_cgi_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                       phot='v', # only using V-band magnitudes as a standard
                       spt=hconf['star']['stellar_type'],
                       pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
    )

    contrast = 1e-5
    nframes, exptime, gain, snr_out, optflag = \
        get_cgi_eetc.calc_exp_time(
            sequence_name=hconf['hardware']['sequence_list'][0],
            snr=1,
            scale=contrast,
            scale_bright=contrast,
        )

    print(nframes, exptime, gain, snr_out, optflag)

    # Generate single image
    im = imager.get_image(dm1_list[0],
                                 dm2_list[0],
                                 exptime,
                                 gain=gain,
                                 crop=crop_corgi,
                                 lind=1,
                                 peakflux=1,
                                 cleanrow=1024,
                                 cleancol=1024,
                                 fixedbp=cstrat.fixedbp,
                                 wfe=None)
    
    return im
    

def test_finite_stellar_diam():
    '''
    This minimal test compares images taken with and without the finite
        stellar diameter model.
    '''
    
    # First, take a single image without the finite stellar diameter model
    t=time.time()
    use_finite_stellar_diam = False
    pt_star_image = simulate_test_image(use_finite_stellar_diam)
    t_point = time.time()-t
    
    # Now, repeat with a minimal model
    t2 =time.time()
    use_finite_stellar_diam = True
    finite_star_image = simulate_test_image(use_finite_stellar_diam)
    t_finite = time.time() - t2
    
    # The image generation should take longer with the finite stellar diameter model
    assert(t_finite > t_point)
    # The peak intensity should be lower with the finite stellar diameter model
    assert(np.max(finite_star_image)<np.max(pt_star_image))
    
if __name__ == '__main__':
    test_finite_stellar_diam()
