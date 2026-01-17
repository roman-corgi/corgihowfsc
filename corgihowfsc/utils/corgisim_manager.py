import numpy as np
from corgisim import scene, instrument

from corgihowfsc.utils.corgisim_utils import (
    _extract_host_properties_from_hconf,
    CGI_TO_CORGI_MAPPING,
    SUPPORTED_CGI_MODES,
    map_wavelength_to_corgisim_bandpass
    )

class CorgisimManager:
    """
    Manages Corgisim optics and scence generation for cgi-howfsc intergration. 

    This class handles: 
    - Mapping CGI modes to corgisim modes 
    - Host star property extraction and management 
    - Scene and Optics config 
    - PSF and detecor image

    """

    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None):
        """
        
        Args:
         cfg: CoronagraphMode object with bandpass information (cfg.sl_list)
         hconf: Host configuration dict/object with star properties
         cor: CGI coronagraph mode (e.g., 'narrowfov', 'nfov_flat', 'nfov_dm')
         corgi_overrides: Optional dict of CorgiSim-specific overrides:
            - bandpass: str, bandpass number ('1', '2', '3', '4')
            - is_noise_free: bool, generate noise-free images (default: True)
            - output_dim: int, output image dimension (default: 51)
            - polaxis: int, polarization axis (default: 10)
            - Vmag: float, override host star V magnitude
            - sptype: str, override spectral type
            - ref_flag: bool, use reference spectrum (default: False)
        """

        if corgi_overrides is None: 
            corgi_overrides = {}
        
        self.cfg = cfg 
        self.cstrat = cstrat
        self.hconf = hconf 
        self.cor = cor 
        self.corgi_overrides = corgi_overrides

        self._validate_inputs()
        
        self._initialize_config()

        self._initialize_base_scene()

    def _validate_inputs(self):
        """Validate required inputs"""
        if self.cor not in SUPPORTED_CGI_MODES:
            raise ValueError(
                f"corgihowfsc backend does not support cor mode '{self.cor}'. "
                f"Supported modes: {SUPPORTED_CGI_MODES}"
            )
        if not hasattr(self.cfg, 'sl_list') or len(self.cfg.sl_list) == 0:
            raise ValueError("cfg.sl_list must contain bandpass information")

    def _initialize_config(self):
        """Initialise the setup for corgisim"""
        if 'bandpass' not in self.corgi_overrides:
            wavelength = self.cfg.sl_list[1].lam
            self.bandpass = map_wavelength_to_corgisim_bandpass(wavelength)
        else:
            self.bandpass = self.corgi_overrides['bandpass']

        # Validate bandpass
        if self.bandpass not in ['1', '2', '3', '4']:
            raise ValueError("bandpass must be one of ['1', '2', '3', '4']")
        
        # map cgihowfsc mode to corgihowfsc
        corgi_base_mode = CGI_TO_CORGI_MAPPING[self.cor]
        self.cor_mapped = f'{corgi_base_mode}_band{self.bandpass}'

        # Extract host star properties
        if self.hconf is not None:
            self.host_star_properties = _extract_host_properties_from_hconf(self.hconf)
        else:
            self.host_star_properties = {
                'Vmag': 2.56,  # default to del Leo
                'spectral_type': 'F0V',
                'ref_flag': 1
            }
        
        # Set other corgihowfsc specific parameters; if not provided, use defaults
        self.is_noise_free = self.corgi_overrides.get('is_noise_free', True)
        self.output_dim = self.corgi_overrides.get('output_dim', 153) # default to match gitl image
        self.polaxis = self.corgi_overrides.get('polaxis', 10)
        self.Vmag = self.corgi_overrides.get('Vmag', self.host_star_properties['Vmag'])
        self.sptype = self.corgi_overrides.get('sptype', self.host_star_properties['spectral_type'])
        self.ref_flag = self.corgi_overrides.get('ref_flag', self.host_star_properties['ref_flag'])
        self._mode = 'excam'  # default camera mode

    def _initialize_base_scene(self):
        # Initialise scene object 
        point_source_info = [] # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(self.host_star_properties, point_source_info)

    def _get_bandpass_recipe(self, lind):
        subband_option = ['a', 'b', 'c']

        if lind < 0 or lind >= len(subband_option):
            raise ValueError(f"lind must be between 0 and {len(subband_option)-1}")
        
        return self.bandpass + subband_option[lind]

    def create_optics(self, dm1v, dm2v, lind):
        bandpass_recipe = self._get_bandpass_recipe(lind)

        optics_keywords = {
            'cor_type': self.cor_mapped,
            'use_errors': 2,
            'polaxis': self.polaxis,
            'output_dim': self.output_dim,
            'use_dm1': 1,
            'dm1_v': dm1v,
            'use_dm2': 1,
            'dm2_v': dm2v,
            'use_fpm': 1,
            'use_lyot_stop': 1,
            'use_field_stop': 1
        }

        optics = instrument.CorgiOptics(
            self._mode,
            bandpass_recipe,
            optics_keywords=optics_keywords,
            if_quiet=True
        )

        return optics
    
    def generate_host_star_psf(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, bias=0):
        optics = self.create_optics(dm1v, dm2v, lind)

        sim_scene = optics.get_host_star_psf(self.base_scene)

        if self.is_noise_free:
            return sim_scene.host_star_image.data
        else:
            # generate detector image
            emccd_dict = {'em_gain': gain, 'bias':bias, 'cr_rate': 0}
            detector = instrument.CorgiDetector(emccd_dict)
            master_dark = self.generate_master_dark(detector, exptime, gain)
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            B = bias * np.ones((self.output_dim, self.output_dim))
            return (sim_scene.image_on_detector.data - B)/gain - master_dark

    def generate_master_dark(self, detector, exptime, gain):
        """
        dark:  master dark
        FPM: fixed pattern noise map
        gain: EM gain
        exptime: exposure time
        D: dark current rate map
        C: CIC map
        """
        D = detector.emccd.dark_current * np.ones((self.output_dim,self.output_dim))
        C = detector.emccd.cic * np.ones((self.output_dim,self.output_dim))
        FPN = np.zeros((self.output_dim,self.output_dim)) # Not included in emccd_detect
        dark = FPN / gain + exptime * D + C


        return dark
    
    def generate_off_axis_psf(self, dm1v, dm2v, dx, dy, companion_vmag=None, lind=0, exptime=1.0, gain=1, bias=0):
        # TODO: move bias to be a class attribute
        if companion_vmag is None:
            companion_vmag = self.Vmag
        
        # Create scene with off-axis point source 
        point_source_info = [
            {
                'Vmag': companion_vmag,
                'magtype': 'vegamag',
                'position_x': dx,
                'position_y': dy
            }
        ]

        optics = self.create_optics(dm1v, dm2v, lind)
        scene_with_offaxis_source = scene.Scene(self.host_star_properties, point_source_info)

        sim_scene = optics.get_host_star_psf(self.base_scene)
        sim_scene = optics.inject_point_sources(scene_with_offaxis_source, sim_scene)

        if self.is_noise_free:
            return sim_scene.point_source_image.data
        else:
            # generate detector image
            emccd_dict = {'em_gain': gain, 'cr_rate': 0, 'bias': bias}
            detector = instrument.CorgiDetector(emccd_dict)
            master_dark = self.generate_master_dark(detector, exptime, gain)
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            B = bias * np.ones((self.output_dim, self.output_dim))
            return (sim_scene.image_on_detector.data - B)/gain - master_dark

