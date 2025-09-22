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
from howfsc.scripts.gitlframes import sim_gitlframe

## import packages
from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
import numpy as np
import proper
from corgisim import outputs
import time

class GitlImage:
    """
    GITL image generator that takes required inputs for cgi-howfsc and can generate images using either cgi-howfsc (compact) or corgisim optical model.
    """
    # Mapping configuration - easy to update for new modes
    CGI_TO_CORGI_MAPPING = {
        'narrowfov': 'hlc',
        'nfov_flat': 'hlc', 
        'nfov_dm': 'hlc'
        # NOTE - Add new mappings here as support is added
        # 'widefov': 'widefov',  # Future support
        # 'spec': 'spec',    # Future spectroscopy mode
    }
    
    SUPPORTED_CGI_MODES = list(CGI_TO_CORGI_MAPPING.keys())
    SUPPORTED_CORGI_MODES = list(set(CGI_TO_CORGI_MAPPING.values()))

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


    def _init_corgihowfsc (self, corgi_overrides):
        """Initialise for corgihowfsc mode. Mapped the input from exisiting cgihowfsc files"""

        # Set default for corgi overrides
        if corgi_overrides is None:
            corgi_overrides = {}

        # map wavelength to corgisim bandpass if not provided
        if 'bandpass' not in corgi_overrides:
            wavelength = self.cfg.sl_list[1].lam
            bandpass = map_wavelength_to_corgisim_bandpass(wavelength)
        else:
            bandpass = corgi_overrides['bandpass']

        # Validate that we support this mode for corgihowfsc mapping
        if self.cor not in self.SUPPORTED_CGI_MODES:
            raise ValueError(
                f"corgihowfsc backend does not support cor mode '{self.cor}'. "
                f"Supported modes: {self.SUPPORTED_CGI_MODES}"
            )
        
        # Map cgihowfsc mode to corgihowfsc 
        corgi_base_mode = self.CGI_TO_CORGI_MAPPING[self.cor]

        # Validate bandpass 
        if bandpass not in ['1', '2', '3', '4']:
            raise ValueError("bandpass must be one of ['1', '2', '3', '4']")
        
        self.bandpass = bandpass

        # Set corongraph mode for corgisim - combine mode with bandpass
        self.cor_mapped = f'{corgi_base_mode}_band{bandpass}'

        # Handle host star properties required by corgisim scene object
        if self.hconf is not None:
            host_star_properties = self._extract_host_properties_from_hconf(self.hconf)
        else:
            host_star_properties = {
                'Vmag': 2.56,  # default to del Leo
                'spectral_type': 'A5V',  # default to del Leo
                'magtype': 'vegamag',  # standard default
                'ref_flag': False  # standard default
            }
        
        # Set other corgihowfsc specific parameters 
        self.is_noise_free = corgi_overrides.get('is_noise_free', True)
        self.output_dim = corgi_overrides.get('output_dim', 51)
        self.polaxis = corgi_overrides.get('polaxis', 10)
        self.Vmag = corgi_overrides.get('Vmag', host_star_properties['Vmag'])
        self.sptype = corgi_overrides.get('sptype', host_star_properties['spectral_type'])
        self.ref_flag = corgi_overrides.get('ref_flag', host_star_properties['ref_flag'])
        self._mode = 'excam'  # default camera mode
        
        # Initialise scene object 
        point_source_info = [] # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(host_star_properties, point_source_info)
    
    @staticmethod
    def _extract_host_properties_from_hconf(hconf):
        """Extract host star properties from hconf object"""
        try:
            star_config = hconf.get('star', {}) if isinstance(hconf, dict) else getattr(hconf, 'star', {})
            
            # Extract stellar properties, preferring target values if available
            Vmag = (star_config.get('stellar_vmag_target') or 
                star_config.get('stellar_vmag') or 
                2.56)  # default fallback
            
            sptype = (star_config.get('stellar_type_target') or 
                    star_config.get('stellar_type') or 
                    'A5V')  # default fallback
            
            return {
                'Vmag': Vmag,
                'spectral_type': sptype,
                'magtype': 'vegamag',  # standard default
                'ref_flag': False  # standard default
            }
        except (AttributeError, KeyError) as e:
            raise ValueError(f"hconf missing required star configuration: {e}")

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
        subband_option = ['a', 'b', 'c'] 
        bandpass_recipe = self.bandpass + subband_option[lind] 
        
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        optics_keywords = {'cor_type': self.cor_mapped, 'use_errors': 2, 'polaxis': self.polaxis, 'output_dim': self.output_dim, \
                           'use_dm1': 1, 'dm1_v': dm1v, 'use_dm2': 1, 'dm2_v': dm2v, 'use_fpm': 1, 'use_lyot_stop': 1,
                           'use_field_stop': 1}

        optics = instrument.CorgiOptics(self._mode, bandpass_recipe, optics_keywords=optics_keywords, if_quiet=True)

        sim_scene = optics.get_host_star_psf(self.base_scene) 

        if self.is_noise_free:
            return sim_scene.host_star_image.data
        else:
            emccd_dict = {'em_gain': gain, 'cr_rate': 0}
            detector = instrument.CorgiDetector(emccd_dict)
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            return sim_scene.image_on_detector.data

    def gitlframe_cgihowfsc(self, dmlist, peakflux, fixedbp, exptime, crop, lind, cleanrow=1024, cleancol=1024):
        """ 
        Generate a GITL frame using the cgi-howfsc optical model. Following the procedure in sim_gitlframe in howfsc.util.gitlframes.
        This is adapted from sim_gitlframe in howfsc.util.gitlframes.
        """

        return sim_gitlframe(self.cfg, dmlist, fixedbp, peakflux, exptime, crop, lind, cleanrow=1024, cleancol=1024)

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

# Helper function to map wavelength to corgisim bandpass
def map_wavelength_to_corgisim_bandpass(wavelength_m, tolerance=3e-9):
    """
    Map wavelength to CorgiSim bandpass label.
    
    Args:
        wavelength_m: Wavelength in meters
        tolerance: Matching tolerance in meters (default ±3nm)
        
    Returns:
        CorgiSim bandpass label ('1', '2', '3', or '4')
    """
    corgisim_wavelengths = {
        '1': 575e-9, '2': 660e-9, '3': 730e-9, '4': 825e-9}
    
    for bandpass, wl in corgisim_wavelengths.items():
        if abs(wavelength_m - wl) <= tolerance:
            return bandpass
    
    available_nm = [wl * 1e9 for wl in corgisim_wavelengths.values()]
    raise ValueError(f"Wavelength {wavelength_m*1e9:.1f} nm does not match any "
                    f"CorgiSim options {available_nm} nm within ±{tolerance*1e9:.0f} nm")




