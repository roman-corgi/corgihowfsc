import numpy as np
from corgisim import scene, instrument

import logging 
log = logging.getLogger(__name__)

from corgihowfsc.utils.corgisim_utils import (
    _extract_host_properties_from_hconf,
    CGI_TO_CORGI_MAPPING,
    SUPPORTED_CGI_MODES,
    map_wavelength_to_corgisim_bandpass, 
    _MANAGER_KEYS
    )
from corgihowfsc.utils import onboard_processing

class CorgisimManager:
    """
    Manages Corgisim optics and scene generation for cgi-howfsc integration. 

    This class handles: 
    - Mapping CGI modes to corgisim modes 
    - Host star property extraction and management 
    - Scene and Optics config 
    - PSF and detector image
    """

    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None, emccd_overrides=None, cosmic_ray_filtering=None):
        """
        Args:
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
            cor: CGI coronagraph mode (e.g., 'narrowfov', 'nfov_flat', 'nfov_dm')
            corgi_overrides: Optional dict of CorgiSim-specific overrides:
                See corgisim doc for details, but some examples include:
                - bandpass: str, bandpass number ('1', '2', '3', '4')
                - is_noise_free: bool, generate noise-free images (default: True)
                - output_dim: int, output image dimension (default: 51)
                - polaxis: int, polarization axis (default: 10)
                - Vmag: float, override host star V magnitude
                - sptype: str, override spectral type
                - ref_flag: bool, use reference spectrum (default: False)
            emccd_overrides: Optional dict of EMCCD-specific overrides:
                See corgisim doc for details, but some examples include:
                - em_gain: float, EM gain setting (default: 1)
                - bias: float, detector bias level (default: 0)
                - cr_rate: float, cosmic ray rate (default: 5)
            cosmic_ray_filtering: Optional dict of cosmic ray filtering parameters:
                - cosmic_filter_width
                - cosmic_saturation_threshold
                - cosmic_plateau_threshold
                - frame_combine
        """

        if corgi_overrides is None: 
            corgi_overrides = {}

        if emccd_overrides is None:
            emccd_overrides = {}

        if cosmic_ray_filtering is None:
            cosmic_ray_filtering = {}
        
        self.cfg = cfg 
        self.cstrat = cstrat
        self.hconf = hconf 
        self.cor = cor 
        self.corgi_overrides = corgi_overrides
        self.emccd_overrides = emccd_overrides
        self.cosmic_ray_filtering = cosmic_ray_filtering
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
            mid_index = len(self.cfg.sl_list) // 2
            wavelength = self.cfg.sl_list[mid_index].lam
            log.info(f"Mapping wavelength {wavelength*1e9:.1f} nm to CorgiSim bandpass...")
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
                'Vmag': 2.25,  # default to del Leo
                'spectral_type': 'O5',
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
        self.k_gain = 8.7 # photo e-/DN, calibrated in TVAC

    def _initialize_emccd_params(self):
        """
        Initialize EMCCD parameters from overrides or defaults.
        Other parameeter can be added here as needed, otherwise they will take the default values from CorgiDetector.
        """
        self.bias = self.emccd_overrides.get('bias', 0) # default should be 1500
        self.cr_rate = self.emccd_overrides.get('cr_rate', 0) # default should be 5

    def _initialize_cosmic_ray_filtering(self):
        # Setup the onboard processing parameters for cosmic ray filtering and frame combination
        self.cosmic_filter_width = self.cosmic_ray_filtering.get('cosmic_filter_width', 2)
        self.cosmic_saturation_threshold = self.cosmic_ray_filtering.get('cosmic_saturation_threshold', 0.99)
        self.cosmic_plateau_threshold = self.cosmic_ray_filtering.get('cosmic_plateau_threshold', 0.85)
        self.frame_combine = self.cosmic_ray_filtering.get('frame_combine', 'mean')

    def _initialize_base_scene(self):
        # Initialise scene object 
        point_source_info = [] # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(self.host_star_properties, point_source_info)

    def _get_bandpass_recipe(self, lind):
        if self.bandpass == '3':
            subband_option = ['a', 'b', 'c', 'd', 'e', 'g'] # band 3 has more subband options, need to update the function to account for this. For now we just default to 'a', 'b', 'c' for all bandpasses but this does not apply to some other bands
        else:
            subband_option = ['a', 'b', 'c']

        if lind < 0 or lind >= len(subband_option):
            raise ValueError(f"lind must be between 0 and {len(subband_option)-1}")
        
        return self.bandpass + subband_option[lind]

    def _get_passthrough_keywords(self, is_corgi_overrides=True):
        """
        Return any corgi_overrides keys that are not manager-level keys, to be
        forwarded directly to CorgiOptics as optics_keywords.
        """
        if is_corgi_overrides:
            return {k: v for k, v in self.corgi_overrides.items() if k not in _MANAGER_KEYS}
        else:
            return {k: v for k, v in self.emccd_overrides.items() if k not in _MANAGER_KEYS}

    def create_emccd_detector(self, gain=None):
        """
        Create a CorgiDetector instance with the specified EMCCD parameters.
        """
        self._initialize_emccd_params()
        
        emccd_dict = {
            'em_gain': gain,
            'bias': self.bias,
            'cr_rate': self.cr_rate
        }

        emccd_dict.update(self._get_passthrough_keywords(is_corgi_overrides=False))  # Update with any additional overrides

        detector = instrument.CorgiDetector(emccd_dict)

        return detector

    def create_optics(self, dm1v, dm2v, lind):
        bandpass_recipe = self._get_bandpass_recipe(lind)

        # Hardcoded defaults
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

        # Merge in any pass-through keywords from corgi_overrides, then
        optics_keywords.update(self._get_passthrough_keywords())

        # re-apply DM voltages so they can never be accidentally overridden.
        optics_keywords['dm1_v'] = dm1v
        optics_keywords['dm2_v'] = dm2v

        optics = instrument.CorgiOptics(
            self._mode,
            bandpass_recipe,
            optics_keywords=optics_keywords,
            if_quiet=True
        )

        return optics

    def generate_on_axis_psf(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, nframes=1):
        """
        Generate the on-axis (host star) PSF with optional detector noise simulation.

        Simulates the host star PSF through the coronagraph optical system with the
        focal plane mask removed. If noise-free mode is active, returns the noiseless
        host star image directly. Otherwise, applies detector effects and returns the
        mean of `nframes` bias- and dark-subtracted, gain-corrected frames.

        Parameters
        ----------
        dm1v : ndarray
            Deformable mirror 1 actuator voltages.
        dm2v : ndarray
            Deformable mirror 2 actuator voltages.
        lind : int, optional
            Wavelength/bandpass index used to select the bandpass recipe.
            Default is 0.
        exptime : float, optional
            Exposure time in seconds for each frame. Default is 1.0.
        gain : float, optional
            EMCCD EM gain. Default is 1.
        nframes : int, optional
            Number of frames to generate and coadd. Default is 1.

        Returns
        -------
        ndarray
            2D array of shape (output_dim, output_dim). In noise-free mode, the
            noiseless host star image in simulation units. In noisy mode, the mean
            of `nframes` bias- and dark-subtracted, gain-corrected frames in electrons.
        """

        bandpass_recipe = self._get_bandpass_recipe(lind)
        use_pupil_mask = 0 if 'hlc' in self.cor_mapped else 1

        optics_keywords = {
            'cor_type': self.cor_mapped,
            'use_errors': 2,
            'polaxis': self.polaxis,
            'output_dim': self.output_dim,
            'use_dm1': 1,
            'dm1_v': dm1v,
            'use_dm2': 1,
            'dm2_v': dm2v,
            'use_fpm': 0,
            'use_lyot_stop': 1,
            'use_field_stop': 0,
            'use_pupil_mask': use_pupil_mask
        }

        optics = instrument.CorgiOptics(
            self._mode,
            bandpass_recipe,
            optics_keywords=optics_keywords,
            if_quiet=True
        )

        sim_scene = optics.get_host_star_psf(self.base_scene)

        if self.is_noise_free:
            return sim_scene.host_star_image.data
        else:
            # generate detector image
            detector = self.create_emccd_detector(gain)

            # initialize cosmic ray filtering parameters
            self._initialize_cosmic_ray_filtering()

            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            master_dark = self.generate_master_dark(detector, exptime)

            # Get the raw frames from the detector
            raw_frames_dn = []
            for n in range(nframes):
                sim_scene = detector.generate_detector_image(sim_scene, exptime)
                raw_frames_dn.append(sim_scene.image_on_detector.data)

            # Apply cosmic ray filtering
            ProcessedFrame = onboard_processing.process_onboard_frames(
                raw_frames_dn,
                bias_e = self.bias,
                e_per_dn = detector.emccd.eperdn,
                em_gain = gain,
                full_well_image_e = detector.emccd.full_well_image,
                full_well_serial_e = detector.emccd.full_well_serial,
                master_dark_e = master_dark,
                cosmic_filter_width = self.cosmic_filter_width,
                saturation_threshold = self.cosmic_saturation_threshold,
                plateau_threshold = self.cosmic_plateau_threshold,
                combine = self.frame_combine)

            filtered_frame = ProcessedFrame.calibrated
            return filtered_frame


    def generate_host_star_psf(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, nframes=1, fixedbp=None):
        """
        Generate the host star PSF using the standard coronagraph configuration.

        Simulates the host star PSF through the coronagraph optical system with the focal plane in. If noise-free mode
        is active, returns the noiseless host star image directly. Otherwise, applies detector effects and returns the
        mean of `nframes` bias- and dark-subtracted, gain-corrected frames.

        Parameters
        ----------
        dm1v : ndarray
            Deformable mirror 1 actuator voltages.
        dm2v : ndarray
            Deformable mirror 2 actuator voltages.
        lind : int, optional
            Wavelength/bandpass index used to select the bandpass recipe.
            Default is 0.
        exptime : float, optional
            Exposure time in seconds for each frame. Default is 1.0.
        gain : float, optional
            EMCCD EM gain. Default is 1.
        nframes : int, optional
            Number of frames to generate and coadd. Default is 1.
        fixedbp : array_like of bool, optional
            Fixed bad-pixel mask forwarded to onboard processing. Defaults to
            no fixed bad pixels.

        Returns
        -------
        ndarray
            2D array of shape (output_dim, output_dim). In noise-free mode, the
            noiseless host star image in simulation units. In noisy mode, the mean
            of `nframes` bias- and dark-subtracted, gain-corrected frames in electrons.

        See Also
        --------
        generate_on_axis_psf : Equivalent method with explicit optics keyword construction.
        """

        optics = self.create_optics(dm1v, dm2v, lind)

        sim_scene = optics.get_host_star_psf(self.base_scene)

        if self.is_noise_free:
            return sim_scene.host_star_image.data
        else:
            # generate detector image
            detector = self.create_emccd_detector(gain)

            # initialize cosmic ray filtering parameters
            self._initialize_cosmic_ray_filtering()

            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            master_dark = self.generate_master_dark(detector, exptime)

            # Get the raw frames from the detector
            raw_frames_dn = []
            for n in range(nframes):
                sim_scene = detector.generate_detector_image(sim_scene, exptime)
                raw_frames_dn.append(sim_scene.image_on_detector.data)

            # Apply cosmic ray filtering
            ProcessedFrame = onboard_processing.process_onboard_frames(
                raw_frames_dn,
                bias_e = self.bias,
                e_per_dn = detector.emccd.eperdn,
                em_gain = gain,
                full_well_image_e = detector.emccd.full_well_image,
                full_well_serial_e = detector.emccd.full_well_serial,
                master_dark_e = master_dark,
                fixed_bp = None, # FIX - it should be a real fixed bad pixel map here
                cosmic_filter_width = self.cosmic_filter_width,
                saturation_threshold = self.cosmic_saturation_threshold,
                plateau_threshold = self.cosmic_plateau_threshold,
                combine = self.frame_combine)

            filtered_frame = ProcessedFrame.calibrated
            return filtered_frame

    def generate_efield(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, bias=0, crop=None):
        """
        Generate the e-field from corgisim
        Args:
            dm1v, dm2v: DM1 and DM2 voltages
            lind: wavelength index
            exptime: exposure time
            crop:  4-tuple of (lower row, lower col, number of rows,
                    number of columns), indicating where in a clean frame a PSF is taken.
                    All are integers; the first two must be >= 0 and the second two must be > 0. Only used if name = 'cgi-howfsc'.
            gain: EM gain setting for the detector model. Defaults to 1.
            bias: Detector bias/offset level [e-]. Defaults to 0.
        Return:
            Generated_efield: Generated electric field, full or cropped. Should be in normalized unit. 
        """
        optics = self.create_optics(dm1v, dm2v, lind)
        generated_efield = optics.get_e_field()

        e_field_norm = np.zeros_like(generated_efield)

        use_pupil_mask = 0 if 'hlc' in self.cor_mapped else 1

        optics.optics_keywords.update({'use_fpm': 0, 'use_lyot_stop': 1, 'use_field_stop': 0, 'use_pupil_mask': use_pupil_mask})  # to get the unocculted e-field for normalization

        e_field_unocc = optics.get_e_field()

        for i in range(len(generated_efield)):
            peak_i = np.sqrt(np.nanmax(np.abs(e_field_unocc[i])**2))
            e_field_norm[i] = generated_efield[i] / peak_i
        
        return e_field_norm


    def generate_master_dark(self, detector, exptime):
        """
        Generate a master dark frame for the EMCCD detector.
        For onboard-processing, we need a subtract bias-subtracted, gain-divided master dark in electrons.
        
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
        log.debug(f"Current detector gain in generate_master_dark: {detector.emccd.em_gain}")        
        dark = FPN / detector.emccd.em_gain + exptime * D + C

        return dark
    
    def generate_off_axis_psf(self, dm1v, dm2v, dx, dy, companion_vmag=None, lind=0, exptime=1.0, gain=1, nframes=1):
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
            detector = self.create_emccd_detector(gain)

            # initialize cosmic ray filtering parameters
            self._initialize_cosmic_ray_filtering()

            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            master_dark = self.generate_master_dark(detector, exptime)

            # Get the raw frames from the detector
            raw_frames_dn = []
            for n in range(nframes):
                sim_scene = detector.generate_detector_image(sim_scene, exptime)
                raw_frames_dn.append(sim_scene.image_on_detector.data)
            
            # Apply cosmic ray filtering
            ProcessedFrame = onboard_processing.process_onboard_frames(
                raw_frames_dn,
                bias_e = self.bias,
                e_per_dn = detector.emccd.eperdn,
                em_gain = gain,
                full_well_image_e = detector.emccd.full_well_image,
                full_well_serial_e = detector.emccd.full_well_serial,
                master_dark_e = master_dark,
                cosmic_filter_width = self.cosmic_filter_width,
                saturation_threshold = self.cosmic_saturation_threshold,
                plateau_threshold = self.cosmic_plateau_threshold,
                combine = self.frame_combine)

            filtered_frame = ProcessedFrame.calibrated
            return filtered_frame