# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.

import os
import numpy as np
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
    def __init__(self, name, cor, polaxis=10, cfg=None, cstrat=None, hconf=None, bandpass=None, Vmag=2.56, sptype='A5V', ref_flag=False, is_noise_free=True, output_dim=51):

        """
        Gitl Image class should be able to handle either corgisim, or calling cgisim, following the defintion from sim_gitlframe in howfsc.util.gitlframes.
        Mode not supported yet: widefov. 

        Arguments:
            name: string, must be either 'corgihowfsc' or 'cgi-howfsc'. An other value will raise an error. 
            cor: coronagraph mode. 
                If name = 'corgihowfsc', this must be 'hlc'. 
                If name = 'cgi-howfsc', this must be one of ['narrowfov', 'nfov_flat', 'nfov_dm']. 
            bandpass: If name = 'corgihowfsc', this must be one of ['1', '2', '3', '4'], corresponding to corgisim bandpass options. 
                If name = 'cgi-howfsc', this is always taken from cfg.sl_list.lam. 
            polaxis: integer, polarization axis setting for the camera.  Must be one of [0, 10, 20, 30].  Default is 10.
            cfg: CoronagraphMode object, only required if name = 'cgi-howfsc'.
            cstrat: ControlStrategy object, only required if name = 'cgi-howfsc'.
            Vmag: float, V magnitude of the host star, default to 2.56 (del Leo).
            sptype: string, spectral type of the host star, default to 'A5V'.
            ref_flag: boolean, if True, use reference spectrum from pysynphot, otherwise use Pickles atlas. Default to False.               
            is_noise_free: boolean, if True, generate a noise-free frame. Default to True. Only used if name = 'corgihowfsc'.
            output_dim: integer, output dimension of the cropped image from corgisim. Default to 51. 
        """

        # Validate the name first
        if name not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("name must be 'corgihowfsc' or 'cgi-howfsc'")

        self.name = name
        self._mode = 'excam'  # default and the only mode we are using for the camera
        self.polaxis = polaxis # TODO - build a check for polaxis?

        # Initialise based on the name
        if self.name == 'corgihowfsc':
            self._init_corgihowfsc(cor, bandpass, Vmag, sptype, ref_flag, is_noise_free, output_dim, hconf)
        else:
            self._init_cgihowfsc(cor, cfg, cstrat, hconf)

    def _init_corgihowfsc (self, cor, bandpass, Vmag, sptype, ref_flag, is_noise_free, output_dim, hconf):
        """Initialise for corgihowfsc mode."""
        # Set coronagraph mode
        self.cor = cor if cor is not None else 'hlc'
        if self.cor != 'hlc':
            raise ValueError("cor must be 'hlc' for corgisim")
            
        # Validate bandpass 
        if bandpass is None or bandpass not in ['1', '2', '3', '4']:
            raise ValueError("bandpass must be one of ['1', '2', '3', '4'] for corgihowfsc and cannot be None")
        self.bandpass = bandpass

        # Set other corgihowfsc specific parameters 
        self.is_noise_free = is_noise_free
        self.output_dim = output_dim
        
        # Handle host star properties required by corgisim scene object
        if hconf is not None:
            host_star_properties = self._extract_host_star_properties(hconf)
        else:
            # Use direct parameters if hconf not provided (for backward compatibility)
            host_star_properties = {'Vmag': Vmag,
                                    'spectral_type': sptype,
                                    'magtype': 'vegamag',
                                    'ref_flag': ref_flag}

        # Initialise scene object 
        point_source_info = [] # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(host_star_properties, point_source_info)
        
    def _init_cgihowfsc(self, cor, cfg, cstrat, hconf):
        """Initialise for cgi-howfsc mode."""
        # Set coronagraph mode
        if cor is None:
            raise ValueError("cor is required for cgi-howfsc or corgihowfsc")
        if cor not in ['narrowfov', 'nfov_flat', 'nfov_dm']:
            raise ValueError("cor must be one of ['narrowfov', 'nfov_flat', 'nfov_dm'] for cgi-howfsc")
        
        self.cor = cor

        # Validate cfg and cstrat
        if cfg is None or cstrat is None:
            raise ValueError("cfg and cstrat are required for cgi-howfsc")
        self.cfg = cfg
        self.cstrat = cstrat

        # bandpass is indeed no needed for cgi-howfsc, as it is taken from cfg.sl_list.lam
        if not hasattr(cfg, 'sl_list') or len(cfg.sl_list) == 0:
            raise ValueError("cfg.sl_list must contain bandpass information")
    
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

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create GitlImage instance from a configuration dictionary. 

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:

        """
        return cls(**config_dict)

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
        if self.name == 'cgi-howfsc':
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
        bandpass_receipe = self.bandpass + subband_option[lind] 
        print(bandpass_receipe, lind)
        
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        optics_keywords = {'cor_type': self.cor, 'use_errors': 2, 'polaxis': self.polaxis, 'output_dim': self.output_dim, \
                           'use_dm1': 1, 'dm1_v': dm1v, 'use_dm2': 1, 'dm2_v': dm2v, 'use_fpm': 1, 'use_lyot_stop': 1,
                           'use_field_stop': 1}

        optics = instrument.CorgiOptics(self._mode, bandpass_receipe, optics_keywords=optics_keywords, if_quiet=True)

        sim_scene = optics.get_host_star_psf(self.base_scene) 

        if self.is_noise_free:
            return sim_scene.host_star_image.data
        else:
            emccd_dict = {'em_gain': gain, 'cr_rate': 0}
            detector = instrument.CorgiDetector(emccd_dict)
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            return sim_scene.image_on_detector.data

    def gitlframe_cgihowfsc(self, cfg, dmlist, peakflux, fixedbp, exptime, crop, lind, cleanrow=1024, cleancol=1024):
        """ 
        Generate a GITL frame using the cgi-howfsc optical model. Following the procedure in sim_gitlframe in howfsc.util.gitlframes.
        This is adapted from sim_gitlframe in howfsc.util.gitlframes.
        """

        return sim_gitlframe(cfg, dmlist, fixedbp, peakflux, exptime, crop, lind, cleanrow=1024, cleancol=1024)

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

        if self.name == 'corgihowfsc':
            return self.gitlframe_corgisim(dm1v, dm2v, fixedbp, exptime, gain, lind, cleanrow, cleancol)
        else:  # cgi-howfsc
            if crop is None:
                raise ValueError("crop parameter is required for cgi-howfsc")
            dmlist = [dm1v, dm2v]
            return self.gitlframe_cgihowfsc(self.cfg, dmlist, peakflux, self.cstrat.fixedbp, exptime, crop, lind, cleanrow, cleancol)

    @classmethod
    def create_equivalent_corgisim(cls, cgi_gitl, **corgi_kwargs):
        """
        Create equivalent CorgiSim instance from CGI-HOWFSC instance, by mapping cgi-howfsc input to corgisim. 

        When you want to run the same simulation through full optical model instead of compact model. 
        """
        if cgi_gitl.name != 'cgi-howfsc':
            raise ValueError("Input must be cgi-howfsc GitlImage instance")
        
        # Extract wavelength and map to CorgiSim bandpass
        wavelength = cgi_gitl.cfg.sl_list[1].lam # central wavelength in meters
        mapped_bandpass = map_wavelength_to_corgisim_bandpass(wavelength)

        # Extract host star properties from hconf and convert to corgisim format
        hconf_to_use = cgi_gitl.hconf if hasattr(cgi_gitl, 'hconf') and cgi_gitl.hconf is not None else {'star': {}}
        host_props = cls._extract_host_properties_from_hconf(hconf_to_use)
        
        # Inherit common parameters, allow overrides
        params = {
            'cor': 'hlc',  # corgihowfsc only supports hlc
            'Vmag': corgi_kwargs.get('Vmag', host_props['Vmag']),
            'sptype': corgi_kwargs.get('sptype', host_props['spectral_type']),
            'ref_flag': corgi_kwargs.get('ref_flag', host_props['ref_flag']),
            'polaxis': corgi_kwargs.get('polaxis', cgi_gitl.polaxis),
            'is_noise_free': corgi_kwargs.get('is_noise_free', True),
            'output_dim': corgi_kwargs.get('output_dim', 51)
        }
        
        return cls('corgihowfsc', bandpass=mapped_bandpass, **params)


# Helper function for mapping between configurations
def map_cgi_to_corgi_config(cgi_config, bandpass=None, output_dim=None, polaxis=None, is_noise_free=None, Vmag=None, sptype=None, ref_flag=None):
    """
    Map cgi-howfsc configuration to corgihowfsc configuration
    
    Args:
        cgi_config: Dictionary with cgi-howfsc configuration
        bandpass: Bandpass for corgisim ('1', '2', '3', '4'). If None, defaults to '1'
        output_dim: Output dimension for corgisim. If None, defaults to 51
        polaxis: Polarization axis. If None, uses value from cgi_config or defaults to 10
        is_noise_free: Noise setting. If None, defaults to True
        
        # Stellar property overrides (optional - will use hconf values if not specified)
        Vmag: V magnitude override. If None, uses value from hconf in cgi_config
        sptype: Spectral type override. If None, uses value from hconf in cgi_config
        ref_flag: Reference flag override. If None, uses value from hconf in cgi_config
        
    Returns:
        Dictionary with corgihowfsc configuration
        
    Note:
        By default, stellar properties (Vmag, sptype, ref_flag) come from hconf in cgi_config.
        You can override individual stellar properties if needed, but this is optional.
    """
    # First validate cgi cor type
    if 'cor' not in cgi_config or cgi_config['cor'] not in ['narrowfov', 'nfov_flat', 'nfov_dm']:
        raise ValueError("cgi_config['cor'] must be one of ['narrowfov', 'nfov_flat', 'nfov_dm']")

    if bandpass is None:
        bandpass = map_wavelength_to_corgisim_bandpass(cgi_config['cfg'].sl_list[1].lam)
    if bandpass not in ['1', '2', '3', '4']:
        raise ValueError("bandpass must be one of ['1', '2', '3', '4'] for corgihowfsc")
    # Build corgi config with clear parameter precedence
    corgi_config = {
        'name': 'corgihowfsc',
        'cor': 'hlc',  # corgihowfsc only supports hlc
        'bandpass': bandpass,
        'output_dim': output_dim if output_dim is not None else 51,
        'polaxis': polaxis if polaxis is not None else cgi_config.get('polaxis', 10),
        'is_noise_free': is_noise_free if is_noise_free is not None else True
    }
    
    # Get stellar properties from hconf (or defaults if no hconf)
    if 'hconf' in cgi_config and cgi_config['hconf'] is not None:
        host_props = GitlImage._extract_host_properties_from_hconf(cgi_config['hconf'])
    else:
        # Use default values if no hconf in cgi_config
        hconf_empty = {'star': {}}
        host_props = GitlImage._extract_host_properties_from_hconf(hconf_empty)
    
    # Apply stellar properties with optional overrides
    corgi_config.update({
        'Vmag': Vmag if Vmag is not None else host_props['Vmag'],
        'sptype': sptype if sptype is not None else host_props['spectral_type'],
        'ref_flag': ref_flag if ref_flag is not None else host_props['ref_flag']
    })
    
    return corgi_config

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




