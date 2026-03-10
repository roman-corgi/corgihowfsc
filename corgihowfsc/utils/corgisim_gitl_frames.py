# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.

import os
import numpy as np
import howfsc
from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check
from howfsc.util.insertinto import insertinto as pad_crop
from howfsc.util.loadyaml import loadyaml
from howfsc.scripts.gitlframes import sim_gitlframe, get_efield_cgihowfsc

## import packages
from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
import numpy as np
import proper
from corgisim import outputs
import time

import multiprocessing
from multiprocessing import Process,Manager,Pool,cpu_count
from itertools import repeat
from functools import partial
import itertools

# import helper functions 
from corgihowfsc.utils.corgisim_utils import _extract_host_properties_from_hconf, CGI_TO_CORGI_MAPPING, SUPPORTED_CORGI_MODES, SUPPORTED_CGI_MODES, map_wavelength_to_corgisim_bandpass

from corgihowfsc.utils.corgisim_manager import CorgisimManager


class GitlImage:
    """
    GITL image generator that takes required inputs for cgi-howfsc and can generate images using either cgi-howfsc (compact) or corgisim optical model.
    """

    def __init__(self, cfg, cstrat, hconf, backend='cgi-howfsc', cor=None, corgi_overrides=None, serial_imaging=True):

        """
        Arguments:
            cfg:
                A Configuration object defining a coronagraph-mode setup for CGI, including wavelength channels (sl_list), deformable mirror states (dmlist), 
                and initial DM settings (initmaps), loaded from a YAML file (cfgfile).
                See https://roman-corgi.github.io/corgihowfsc/cfg_docs.html for more details.
            cstrat: 
                A ControlStrategy object which contains the necessary information to perform wavefront sensing and control. 
                See https://roman-corgi.github.io/corgihowfsc/cstrat_docs.html for more details.
                
            hconf:
                A HardwareConfig object that contains instrument configurations and host star properties.
                See https://roman-corgi.github.io/corgihowfsc/hconf_docs.html for more details.

            backend: str, either 'corgihowfsc' or 'cgi-howfsc', indicating which optical model to use for image generation. Default is 'cgi-howfsc'. 
            # TODO - we should change this attribute to something else ... it's not really a backend ... 
            cor: str, CGI coronagraph mode (e.g., 'narrowfov', 'nfov_flat', 'nfov_dm'). 
                Required if backend is 'cgi-howfsc'. Ignored if backend is 'corgihowfsc' since corgisim will use its own internal mapping.

            corgi_overrides: Optional dict of CorgiSim-specific overrides:
                See corgisim doc for details, but some examples include:
                - bandpass: str, bandpass number ('1', '2', '3', '4')
                - is_noise_free: bool, generate noise-free images (default: True)
                - output_dim: int, output image dimension (default: 51)
                - polaxis: int, polarization axis (default: 10)
                - Vmag: float, override host star V magnitude
                - sptype: str, override spectral type
                - ref_flag: bool, use reference spectrum (default: False)

            serial_imaging: Optional, bool
                If true, get_images is performed in serial. If false, get_images is performed in parallel.
        """
        # Validate backend choice
        if backend not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("backend must be 'corgihowfsc' or 'cgi-howfsc'")

        # Store configuration
        self.backend = backend
        self.cfg = cfg 
        self.cstrat = cstrat
        self.hconf = hconf
        self.cor = cor
        self.serial_imaging = serial_imaging
        
        if cor is None:
            raise ValueError("cor mode must be provided")

        # check cgi-howfsc specific inputs
        if cfg is None or cstrat is None: 
            raise ValueError("cfg and cstrat are required") 

        if not hasattr(cfg, 'sl_list') or len(cfg.sl_list) == 0:
            raise ValueError("cfg.sl_list must contain bandpass information")

        # Backend specific initialisation
        if self.backend == 'corgihowfsc':
            self._init_corgihowfsc(corgi_overrides)

        else:
            self.nrow = 153
            self.ncol = 153
            self.lrow = 436
            self.lcol = 436


    def _init_corgihowfsc (self, corgi_overrides):
        """Initialise for corgihowfsc mode. Mapped the input from exisiting cgihowfsc files"""

        self.corgisim_manager = CorgisimManager(self.cfg, self.cstrat, self.hconf, self.cor, corgi_overrides=corgi_overrides)

    def check_gitlframeinputs(self, dm1v, dm2v, fixedbp, exptime, crop, cleanrow, cleancol):
        """Input validation for both simulators."""
        # DM array validation
        for dm in [dm1v, dm2v]:
            check.twoD_array(dm, 'dm', TypeError)
            if dm.shape != (48, 48):
                raise TypeError('DM dimensions must be (48, 48)')

        # Common validation
        check.twoD_array(fixedbp, 'fixedbp', TypeError)
        if fixedbp.shape != (cleanrow, cleancol):
            raise TypeError('fixedbp must match clean frame dimensions')
        if fixedbp.dtype != bool:
            raise TypeError('fixedbp must be boolean')
        check.real_positive_scalar(exptime, 'exptime', TypeError)
            
        # CGI-HOWFSC specific validation
        if self.backend == 'cgi-howfsc':
            if not isinstance(crop, tuple) or len(crop) != 4:
                raise TypeError('crop must be a 4-tuple for cgi-howfsc')
            
            for i, val in enumerate(crop):
                if i < 2:
                    check.nonnegative_scalar_integer(val, f'crop[{i}]', TypeError)
                else:
                    check.positive_scalar_integer(val, f'crop[{i}]', TypeError)
        check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
        check.positive_scalar_integer(cleancol, 'cleancol', TypeError)
        
    def gitlframe_corgisim(self, dm1v, dm2v, fixedbp, exptime, crop, lind=0, gain=1, cleanrow=1024, cleancol=1024, wfe=None):
        """
        Generate a GITL frame using the CorgiSim optical model. Following the procedure in sim_gitlframe in howfsc.util.gitlframes.

        Arguments:
         dm1v: ndarray, absolute voltage map for DM1.
         dm2v: ndarray, absolute voltage map for DM2.
         fixedbp: this is a fixed bad pixel map for a clean frame, and will be a 2D
          boolean array with the same size as a cleaned frame.
         exptime: Exposure time used when collecting the data in in.  Should be a
          real scalar > 0. If is_noise_free = True, this can be any positive value.
         crop: 4-tuple of (lower row, lower col, number of rows, number of cols). Currently not in used
         lind = 0: integer >= 0 indicating which wavelength channel in use. 

        wfe is a placeholder argument for now to keep the option of passing additional wavefront error (e.g. zernike coefficients) to modify the frame generation
        """        
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        return self.corgisim_manager.generate_host_star_psf(dm1v, dm2v, lind=lind, exptime=exptime, gain=gain)

    def gitlframe_cgihowfsc(self, dmlist, peakflux, fixedbp, exptime, crop, lind, cleanrow=1024, cleancol=1024):
        """ 
        Generate a GITL frame using the cgi-howfsc optical model. Following the procedure in sim_gitlframe in howfsc.util.gitlframes.
        This is adapted from sim_gitlframe in howfsc.util.gitlframes.
        """

        return sim_gitlframe(self.cfg, dmlist, fixedbp, peakflux, exptime, crop, lind, cleanrow=1024, cleancol=1024)

    def gitlefield_cgihowfsc(self, dmlist, crop, lind, cleanrow, cleancol):
        """
        Generate a GITL efield using the cgi-howfsc optical model.
        """

        # Creation of dummy variables to satisfy the strict signature of sim_gitlframe

        dummy_bp = np.zeros((cleanrow, cleancol), dtype=bool)

        return get_efield_cgihowfsc(
            cfg=self.cfg,
            dmlist=dmlist,
            fixedbp=dummy_bp,  # For checking
            peakflux=1.0,
            exptime=1.0,
            crop=crop,
            lind=lind,
            cleanrow=cleanrow,
            cleancol=cleancol
        )

    def get_images(self, dm1_list, dm2_list, exptime_list, gain_list, croplist, cstrat, hconf, normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix):
        """
        Get a batch of simulated GITL frames across all wavelength channels and DM configurations, using either the
        corgisim or cgi-howfsc optical model.

        If self.serial_imaging is true:
        Iterates over each wavelength channel in cfg.sl_list and ndm DM configurations per channel, calling get_image
        for each combination. Peakflux is computed per wavelength channel via the normalization strategy.

        If self.serial_imaging is false:
        Images for all wavelength channels and all DM commands are acquired in parallel.

        After each frame is generated, a random bad pixel mask scaled by fracbadpix is applied by setting affected
        pixels to NaN.

        Arguments:
         dm1_list: list of ndarray
          Absolute voltage maps for DM1, one per (wavelength channel, DM config) pair.
          Expected length is len(cfg.sl_list) * ndm, indexed as indj*ndm + indk.
         dm2_list: list of ndarray
          Absolute voltage maps for DM2. Same shape and indexing convention as dm1_list.
         exptime_list: list of float
          Exposure times in seconds for each frame. Same length and indexing as dm1_list.
          Each value must be a real scalar > 0.
         gain_list: list of float
          EM gain settings for each frame. Same length and indexing as dm1_list.
          Each value must be a real scalar >= 1.
         croplist: list of 4-tuple of int
          Crop parameters per wavelength channel, one entry per element of cfg.sl_list.
          Each tuple is (lower row, lower col, number of rows, number of columns);
          the first two values must be >= 0 and the last two must be > 0.
         cstrat: object
          Control strategy object. Must have a fixedbp attribute (ndarray of bool)
          representing the fixed bad pixel map passed to each get_image call.
         hconf: object
          Hardware configuration object passed to normalization_strategy.calc_flux_rate.
         normalization_strategy: object
          Normalization strategy employed for this setup returning (_, peakflux). Called once per wavelength channel.
         get_cgi_eetc: object
          EETC instance passed to normalization_strategy.calc_flux_rate for flux rate calculations.
         ndm: int
          Number of DM configurations (frames) to collect per wavelength channel. This includes probed and unprobed
          commands.
         cfg: object
          Configuration object with a sl_list attribute defining the wavelength channels
          to iterate over.
         fracbadpix: float
          Fraction of pixels to randomly mask as bad (set to NaN) in each frame.
          Must be in [0, 1].

        Returns:
         list of ndarray
          Simulated detector frames in order of (wavelength channel, DM config),
          with random bad pixels set to NaN. Total length is len(cfg.sl_list) * ndm.
        """

        if self.serial_imaging:
            framelist = self.get_images_serial(dm1_list, dm2_list, exptime_list, gain_list, croplist, cstrat, hconf,
                                                normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix)
        else:
            framelist = self.get_images_parallel(dm1_list, dm2_list, exptime_list, gain_list, croplist, cstrat, hconf,
                                                 normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix)

        return framelist

    def get_images_parallel(self, dm1_list, dm2_list, exptime_list, gain_list, croplist, cstrat, hconf,
                                  normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix):
        
        # TODO: Any camera settings or anything else missing?
        
        # Set up the wavelength indices
        ind_list = []
        ind_list_all = []
        for indj, sl in enumerate(cfg.sl_list):
            ind_list.append(indj)
            repeated_indj = [indj] * len(dm1_list)
            ind_list_all.append(repeated_indj)
        ind_list_all = np.ravel(ind_list_all)
        
        # Set up parameter arrays to allow for each wavelength
        dm1_list_all = dm1_list * len(ind_list)
        dm2_list_all = dm2_list * len(ind_list)
        exptime_list_all = exptime_list * len(ind_list)
        gain_list_all = gain_list * len(ind_list)
        crop_list_all = croplist * len(ind_list)
        
        

        # Set up multiprocessing
        manager = multiprocessing.Manager()
        framelist = manager.list()
        
        pool = multiprocessing.Pool(processes=4)
        pool.starmap_async(self.get_image_parallel,[framelist,cstrat,hconf,normalization_strategy,get_cgi_eetc,ndm,cfg,fracbadpix,dm1_list[0],dm2_list[0],
                                                    zip(dm1_list_all,dm2_list_all,exptime_list_all,gain_list_all,crop_list_all,ind_list_all)])        
        
        
    def get_image_parallel(self, framelist,cstrat,hconf,normalization_strategy,get_cgi_eetc,ndm,cfg, fracbadpix, dm1_0, dm2_0,
                           dm1v, dm2v, exptime, gain, crop, lind, 
                           cleanrow=1024, cleancol=1024, fixedbp=np.zeros((1024, 1024), dtype=bool), wfe=None):
        """
        Get a simulated GITL frame using either the corgisim or cgi-howfsc optical model.
        This method is designed to be compatible with both backends.

        Arguments:
         framelist: list of ndarrays
          List of simulated detector frames.
         dm1v: ndarray
          Absolute voltage map for DM1.
         dm2v: ndarray
          Absolute voltage map for DM2.
         exptime: float
          Exposure time in seconds. Must be a real scalar > 0. If is_noise_free=True,
          any positive value is acceptable.
         gain: float, optional
          EM gain setting for the EMCCD. Real scalar >= 1. Default is 1.
         crop: 4-tuple of int, optional
          (lower row, lower col, number of rows, number of columns) defining the
          sub-region of a clean frame from which the PSF is extracted. The first two
          values must be >= 0 and the last two must be > 0. Required when backend is
          'cgi-howfsc'; ignored otherwise.
         lind: int, optional
          Index >= 0 indicating which wavelength channel in cfg to use. Must be less
          than the length of cfg.sl_list. Default is 0.
         peakflux: float, optional
          Peak flux scaling factor. Default is 1. Only used when backend is 'cgi-howfsc'.
         cleanrow: int, optional
          Number of rows in a clean frame. Must be > 0. Defaults to 1024, matching
          the EXCAM detector active area; this should not need to change under nominal
          conditions.
         cleancol: int, optional
          Number of columns in a clean frame. Must be > 0. Defaults to 1024, matching
          the EXCAM detector active area; this should not need to change under nominal
          conditions.
         fixedbp: ndarray of bool, optional
          Fixed bad pixel map of shape (cleanrow, cleancol). Defaults to an array of
          all False (no bad pixels). If None is passed, defaults to an all-False array
          of shape (cleanrow, cleancol).
         wfe: optional
          Placeholder for future support of additional wavefront error inputs (e.g.
          Zernike coefficients). Currently unused.

        Returns:
         list
          Simulated detector frames.

        Raises:
         ValueError
          If crop is None when backend is 'cgi-howfsc', or if other input validation
          checks in check_gitlframeinputs fail.
        """
       
        # TODO: what are the correct camera settings here?
        _, peakflux = normalization_strategy.calc_flux_rate(get_cgi_eetc, hconf, lind, dm1_0, dm2_0, gain=1)
        if fixedbp is None:
            fixedbp = np.zeros((cleanrow, cleancol), dtype=bool)

        # Input validation
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime, crop, cleanrow, cleancol)

        if self.backend == 'corgihowfsc':
            f = self.gitlframe_corgisim(dm1v, dm2v, fixedbp, exptime, gain, lind, cleanrow, cleancol)
        else:  # cgi-howfsc
            if crop is None:
                raise ValueError("crop parameter is required for cgi-howfsc")
            dmlist = [dm1v, dm2v]
            
            f = self.gitlframe_cgihowfsc(dmlist, peakflux, self.cstrat.fixedbp, exptime, crop, lind, cleanrow, cleancol)

        rng = np.random.default_rng(12345)
        bpmeas = rng.random(f.shape) > (1 - fracbadpix)
        f[bpmeas] = np.nan
        framelist.append(f)
        
    def get_images_serial(self, dm1_list, dm2_list, exptime_list, gain_list, croplist, cstrat, hconf,
                                normalization_strategy, get_cgi_eetc, ndm, cfg, fracbadpix):
        rng = np.random.default_rng(12345)
        framelist = []
        for indj, sl in enumerate(cfg.sl_list):
            crop = croplist[indj]
            # TODO: what are correct camera settings here?
            _, peakflux = normalization_strategy.calc_flux_rate(get_cgi_eetc, hconf, indj, dm1_list[0], dm2_list[0], gain=1)
            for indk in range(ndm):
                f = self.get_image(dm1_list[indj*ndm + indk],
                                 dm2_list[indj*ndm + indk],
                                 exptime_list[indj*ndm + indk],
                                 gain=gain_list[indj*ndm + indk],
                                 crop=crop,
                                 lind=indj,
                                 peakflux=peakflux,
                                 cleanrow=1024,
                                 cleancol=1024,
                                 fixedbp=cstrat.fixedbp,
                                 wfe=None)

                bpmeas = rng.random(f.shape) > (1 - fracbadpix)
                f[bpmeas] = np.nan
                framelist.append(f)
                # pass
            # pass
        return framelist


    def get_image(self, dm1v, dm2v, exptime, gain=1, crop=None, lind=0, peakflux=1, cleanrow=1024, cleancol=1024, fixedbp=np.zeros((1024, 1024), dtype=bool), wfe=None):
        """
        Get a simulated GITL frame using either the corgisim or cgi-howfsc optical model.
        This method is designed to be compatible with both backends.

        Arguments:
         dm1v: ndarray
          Absolute voltage map for DM1.
         dm2v: ndarray
          Absolute voltage map for DM2.
         exptime: float
          Exposure time in seconds. Must be a real scalar > 0. If is_noise_free=True,
          any positive value is acceptable.
         gain: float, optional
          EM gain setting for the EMCCD. Real scalar >= 1. Default is 1.
         crop: 4-tuple of int, optional
          (lower row, lower col, number of rows, number of columns) defining the
          sub-region of a clean frame from which the PSF is extracted. The first two
          values must be >= 0 and the last two must be > 0. Required when backend is
          'cgi-howfsc'; ignored otherwise.
         lind: int, optional
          Index >= 0 indicating which wavelength channel in cfg to use. Must be less
          than the length of cfg.sl_list. Default is 0.
         peakflux: float, optional
          Peak flux scaling factor. Default is 1. Only used when backend is 'cgi-howfsc'.
         cleanrow: int, optional
          Number of rows in a clean frame. Must be > 0. Defaults to 1024, matching
          the EXCAM detector active area; this should not need to change under nominal
          conditions.
         cleancol: int, optional
          Number of columns in a clean frame. Must be > 0. Defaults to 1024, matching
          the EXCAM detector active area; this should not need to change under nominal
          conditions.
         fixedbp: ndarray of bool, optional
          Fixed bad pixel map of shape (cleanrow, cleancol). Defaults to an array of
          all False (no bad pixels). If None is passed, defaults to an all-False array
          of shape (cleanrow, cleancol).
         wfe: optional
          Placeholder for future support of additional wavefront error inputs (e.g.
          Zernike coefficients). Currently unused.

        Returns:
         ndarray
          Simulated detector frame.

        Raises:
         ValueError
          If crop is None when backend is 'cgi-howfsc', or if other input validation
          checks in check_gitlframeinputs fail.
        """

        if fixedbp is None:
            fixedbp = np.zeros((cleanrow, cleancol), dtype=bool)

        # Input validation
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime, crop, cleanrow, cleancol)

        if self.backend == 'corgihowfsc':
            return self.gitlframe_corgisim(dm1v, dm2v, fixedbp, exptime, gain, lind, cleanrow, cleancol)
        else:  # cgi-howfsc
            if crop is None:
                raise ValueError("crop parameter is required for cgi-howfsc")
            dmlist = [dm1v, dm2v]
            
            return self.gitlframe_cgihowfsc(dmlist, peakflux, self.cstrat.fixedbp, exptime, crop, lind, cleanrow, cleancol)

    def get_efield(self, dm1v, dm2v, lind=0, crop=None, output_shape=(153, 153),  cleanrow = 1024, cleancol = 1024):
        """
        Get a simulated GITL efield using either corgisim or cgi-howfsc repo's optical model. This get_efield method should be compatible with both cgi-howfsc and corgisim.
        Arguments:
         dm1v: ndarray, absolute voltage map for DM1.
         dm2v: ndarray, absolute voltage map for DM2.
         crop: 4-tuple of (lower row, lower col, number of rows,
          number of columns), indicating where in a clean frame a PSF is taken.
          All are integers; the first two must be >= 0 and the second two must be > 0. Only used if name = 'cgi-howfsc'.
        """

        if crop is None:
            raise ValueError("crop parameter is required for cgi-howfsc")

        if self.backend == 'corgihowfsc':
            return self.corgisim_manager.generate_e_field(dm1v, dm2v, lind)

        else:  # cgi-howfsc
            dmlist = [dm1v, dm2v]
            return self.gitlefield_cgihowfsc(
                dmlist=dmlist,
                crop=crop,
                lind=lind,
                cleanrow=cleanrow,
                cleancol=cleancol
            )

