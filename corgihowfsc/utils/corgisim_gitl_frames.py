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

# import helper functions 
from corgihowfsc.utils.corgisim_utils import _extract_host_properties_from_hconf, CGI_TO_CORGI_MAPPING, SUPPORTED_CORGI_MODES, SUPPORTED_CGI_MODES, map_wavelength_to_corgisim_bandpass

from corgihowfsc.utils.corgisim_manager import CorgisimManager


class GitlImage:
    """
    GITL image generator that takes required inputs for cgi-howfsc and can generate images using either cgi-howfsc (compact) or corgisim optical model.
    """

    def __init__(self, cfg, cstrat, hconf, backend='cgi-howfsc', cor=None, corgi_overrides=None):

        """
        Arguments:
            cor: mode string, only required when name = 'cgi-howfsc'. Must be one of ['narrowfov', 'nfov_flat', 'nfov_dm']. If name = 'corgihowfsc', this will be mapped. 
            polaxis: integer, polarization axis setting for the camera.  Must be one of [0, 10, 20, 30].  Default is 10.
            cfg: CoronagraphMode object, only required if name = 'cgi-howfsc'.
            cstrat: ControlStrategy object, only required if name = 'cgi-howfsc'.
            Vmag: float, V magnitude of the host star, default to 2.56 (del Leo).
            sptype: string, spectral type of the host star, default to 'A5V'.
            ref_flag: boolean, if True, use reference spectrum from pysynphot, otherwise use Pickles atlas. Default to False.               
            is_noise_free: boolean, if True, generate a noise-free frame. Default to True. Only used if name = 'corgihowfsc'.
            output_dim: integer, output dimension of the cropped image from corgisim. Default to 51. 
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
    def get_image(self, dm1v, dm2v, exptime, gain=1, crop=None, lind=0, peakflux=1, cleanrow=1024, cleancol=1024, fixedbp=np.zeros((1024, 1024), dtype=bool), wfe=None):

        """
        Get a simulated GITL frame using either corgisim or cgi-howfsc repo's optical model. This get_image method should be compatible with both cgi-howfsc and corgisim. 
        Arguments:
         dm1v: ndarray, absolute voltage map for DM1.
         dm2v: ndarray, absolute voltage map for DM2.
         mode: coronagraph mode. Currently taking from args.mode, which will be 'narrowfov' for cgi-howfsc loop. If calling corgisim, this will be fixed to 'hlc' for now. 
         bandpass: depending on which mode we are in, it can be a instance object (passing as cfg.sl) or string if we are using corgisim (e.g. '1a', '1b', etc.)
         lind: integer >= 0 indicating which wavelength channel in cfg to use.
          Must be < the length of cfg.sl_list. Only used if name = 'cgi-howfsc'.
         peakflux: float 
         crop: 4-tuple of (lower row, lower col, number of rows,
          number of columns), indicating where in a clean frame a PSF is taken.
          All are integers; the first two must be >= 0 and the second two must be > 0. Only used if name = 'cgi-howfsc'.
         polaxis: integer, polarization axis setting for the camera.  Must be one of [0, 10, 20, 30].  Default is 10.
         cleanrow: Number of rows in a clean frame.  Integer > 0.  Defaults to 1024, the number of active area rows on the EXCAM detector; under nominal conditions, there should be no reason to use anything else.

         exptime: Exposure time used when collecting the data in in. Should be a real scalar > 0 when noise is included. If is_noise_free = True, this can be any positive value.
         gain: EM gain setting for the EMCCD.  Real scalar >= 1.

        wfe is a placeholder argument for now to keep the option of passing additional wavefront error (e.g. zernike coefficients) to modify the frame generation
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

