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

class CorgisimManager:
    """
    Manages Corgisim optics and scene generation for cgi-howfsc integration. 

    This class handles: 
    - Mapping CGI modes to corgisim modes 
    - Host star property extraction and management 
    - Scene and Optics config 
    - PSF and detector image
    """

    def __init__(self, cfg, cstrat, hconf, cor=None, corgi_overrides=None):
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
                'spectral_type': '05',
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

    def _get_passthrough_keywords(self):
        """
        Return any corgi_overrides keys that are not manager-level keys, to be
        forwarded directly to CorgiOptics as optics_keywords.
        """
        return {k: v for k, v in self.corgi_overrides.items() if k not in _MANAGER_KEYS}

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

    def generate_on_axis_psf(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, nframes=1, bias=0):
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
        bias : float, optional
            Detector bias level. Default is 0.

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
            emccd_dict = {'em_gain': gain, 'bias':bias, 'cr_rate': 0}
            detector = instrument.CorgiDetector(emccd_dict)
            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            master_dark = self.generate_master_dark(detector, exptime, gain)
            B = bias * np.ones((self.output_dim, self.output_dim))

            coadd = np.zeros((self.output_dim, self.output_dim))
            for n in range(nframes):
                sim_scene = detector.generate_detector_image(sim_scene, exptime)
                frame = (self.k_gain * sim_scene.image_on_detector.data - B) / gain - master_dark
                coadd += frame
            # frame = (sim_scene.image_on_detector.data - B) * self.k_gain / gain - master_dark
            return coadd/nframes



    def generate_host_star_psf(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, nframes=1, bias=0):
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
        bias : float, optional
            Detector bias level in ADU. Default is 0.

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
            emccd_dict = {'em_gain': gain, 'bias':bias, 'cr_rate': 0}
            detector = instrument.CorgiDetector(emccd_dict)
            # sim_scene.image_on_detector.data is not gain corrected or bias subtracted
            master_dark = self.generate_master_dark(detector, exptime, gain)
            B = bias * np.ones((self.output_dim, self.output_dim))

            coadd = np.zeros((self.output_dim, self.output_dim))
            for n in range(nframes):
                sim_scene = detector.generate_detector_image(sim_scene, exptime)
                frame = (self.k_gain * sim_scene.image_on_detector.data - B) / gain - master_dark
                coadd += frame
            # frame = (sim_scene.image_on_detector.data - B) * self.k_gain / gain - master_dark
            return coadd/nframes

    def generate_efield(self, dm1v, dm2v, lind=0, exptime=1.0, gain=1, bias=0, crop = None):
        """
        Generate the e-field from corgisim
        Args:
            dm1v, dm2v: DM1 and DM2 voltages
            lind: wavelength index
            exptime: exposure time
            crop:  4-tuple of (lower row, lower col, number of rows,
                    number of columns), indicating where in a clean frame a PSF is taken.
                    All are integers; the first two must be >= 0 and the second two must be > 0. Only used if name = 'cgi-howfsc'.
        Return:
            Generated_efield: Generated electric field, full or cropped
        """
        optics = self.create_optics(dm1v, dm2v, lind)
        generated_efield = optics.get_e_field()
        return generated_efield


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
            return (self.k_gain*sim_scene.image_on_detector.data - B)/gain - master_dark

