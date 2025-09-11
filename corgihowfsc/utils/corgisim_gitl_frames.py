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
    def __init__(self, name, cor, bandpass, polaxis=10, cfg=None, cstrat=None, Vmag=2.56, sptype='A5V', ref_flag=False, is_noise_free=True, output_dim=51):
        
        """
        Gitl Image class should be able to handle either corgisim, or calling cgisim, following the defintion from sim_gitlframe in howfsc.util.gitlframes

        mode not supported yet: widefov

        bandpass: depending on which mode we are in, it can be a instance object (passing as cfg.sl) or string if we are using corgisim (e.g. '1a', '1b', etc.)

        """
        # common configs
        self._mode = 'excam'  # default and the only mode we are using for the camera
        self.polaxis = polaxis # polarization axis setting for the camera.  Must be one of [0, 10, 20, 30].  Default is 10.

        # name check for corgisim or cgi-howfsc 
        if name not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("name must be 'corgihowfsc' or 'cgi-howfsc'")
        else:
            self.name = name

        # mode check and mapping 
        allowed_mode = ['hlc','narrowfov', 'nfov_flat', 'nfov_dm']
        if self.name == 'corgihowfsc':
            if cor not in allowed_mode:
                raise ValueError(f"cor must be one of {allowed_mode} for corgisim")
            else: 
                self.cor = 'hlc'

        if self.name == 'cgi-howfsc':
            if cor not in allowed_mode:
                raise ValueError(f"mode must be one of {allowed_mode} for cgi-howfsc")
            else:
                self.cor = cor

        # cgi-howfsc configs
        self.cfg = cfg 
        self.cstrat = cstrat

        # bandpass check 
        if self.name == 'corgihowfsc': 
            corgi_bandpass_option = ['1', '2', '3', '4'] # corgisim bandpass options
            if bandpass not in corgi_bandpass_option:
                raise ValueError(f"bandpass must be one of {corgi_bandpass_option} for corgisim")

        if self.name == 'cgi-howfsc':
            print('Using cgi-howfsc bandpass check ...')
            bandpass_option = [575e-9, 660e-9, 730e-9, 825e-9] # central wvl for 4 bands
            corgi_bandpass_option = ['1', '2', '3', '4'] # corgisim bandpass options

            _bandpass = cfg.sl_list[1].lam # Hardcoded index for the central wavelength for cgi-howfsc 

            tol = 3e-9 # match within +/- 3 nm -> 3e-9 m
            match = [bp for bp in bandpass_option if abs(_bandpass - bp) <= tol]

            if not match:
                nm_opts = [bp * 1e9 for bp in bandpass_option]
                raise ValueError(f"bandpass {_bandpass*1e9:.1f} nm does not match any allowed options {nm_opts} nm within +/- {tol*1e9} nm")

            # map to corgisim bandpass label
            bandpass = corgi_bandpass_option[bandpass_option.index(match[0])]

        self.bandpass = bandpass # map this back to corgisim 

        # host star properties (shared between corgihowfsc and cgi-howfsc)
        host_star_properties = {'Vmag': Vmag,
                                'spectral_type': sptype,
                                'magtype': 'vegamag',
                                'ref_flag': ref_flag}

        # corgisim configs
        self.is_noise_free = is_noise_free # corgisim only 
        self.output_dim = output_dim

        # corgisim configs
        point_source_info = [] # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(host_star_properties, point_source_info)

    def check_gitlframeinputs(self, dm1v, dm2v, fixedbp, exptime, crop, cleanrow, cleancol): 
        """ 
        Input checks for gitlframe generation functions. Orginally from sim_gitlframe in howfsc.util.gitlframes. 
        Common for both cgi-howfsc and corgisim implementations. 

        Arguments:
         dm1v: ndarray, absolute voltage map for DM1.
         dm2v: ndarray, absolute voltage map for DM2.
         fixedbp: this is a fixed bad pixel map for a clean frame, and will be a 2D
          boolean array with the same size as a cleaned frame.
         exptime: Exposure time used when collecting the data in in.  Should be a
          real scalar > 0.
         crop: 4-tuple of (lower row, lower col, number of rows,
          number of columns), indicating where in a clean frame a PSF is taken.
          All are integers; the first two must be >= 0 and the second two must be
          > 0.
         cleanrow: Number of rows in a clean frame.  Integer > 0.  Defaults to
          1024, the number of active area rows on the EXCAM detector; under nominal
          conditions, there should be no reason to use anything else.
         cleancol: Number of columns in clean frame. Integer > 0.  Defaults to
          1024, the number of active area cols on the EXCAM detector; under nominal
          conditions, there should be no reason to use anything else.
        """
        # Check inputs
        for dm in [dm1v, dm2v]:
            check.twoD_array(dm, 'dm', TypeError)
            nact = 48
            if dm.shape != (nact, nact):
                raise TypeError('DM dimensions do not match model')

        check.twoD_array(fixedbp, 'fixedbp', TypeError)
        if fixedbp.shape != (cleanrow, cleancol):
            raise TypeError('fixedbp must be the same size as a cleaned frame')
        if fixedbp.dtype != bool:
            raise TypeError('fixedbp must be boolean')

        if self.name not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("name must be 'corgihowfsc' or 'cgi-howfsc'")
        elif self.name == 'corgihowfsc':
            # do corgisim check - verify if we need test unique to corgisim
            pass 

        elif self.name == 'cgi-howfsc':
            print('Using cgi-howfsc input checks ...')
            # do cgi-howfsc check 
            check.real_positive_scalar(exptime, 'exptime', TypeError)

            if not isinstance(crop, tuple):
                raise TypeError('crop must be a tuple')
            if len(crop) != 4:
                raise TypeError('crop must be a 4-tuple')

            check.nonnegative_scalar_integer(crop[0], 'crop[0]', TypeError)
            check.nonnegative_scalar_integer(crop[1], 'crop[1]', TypeError)
            check.positive_scalar_integer(crop[2], 'crop[2]', TypeError)
            check.positive_scalar_integer(crop[3], 'crop[3]', TypeError)

            check.real_positive_scalar(exptime, 'exptime', TypeError)
            check.real_positive_scalar(peakflux, 'peakflux', TypeError)

            check.nonnegative_scalar_integer(lind, 'lind', TypeError)

            if lind >= len(cfg.sl_list):
                raise ValueError('lind must be < len(cfg.sl_list)')

            check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
            check.positive_scalar_integer(cleancol, 'cleancol', TypeError)

    def gitlframe_corgisim(self, dm1v, dm2v, fixedbp, exptime, crop, lind=0, gain=1, cleanrow=1024, cleancol=1024, wfe=None):
        """
        Generate a GITL frame using the CorgiSim optical model.
        lind: subband index, default to 0 since corgisim only has one subband per bandpass
        """
        subband_option = ['a', 'b', 'c'] 
        bandpass_receipe = self.bandpass + subband_option[lind] # default to 'a' subband for now

        print(f'Bandpass used for corgisim: {bandpass_receipe}')
        
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        # TODO: update this to be optics_keywords - newer version of corgisim will call this optics keywords 
        optics_keywords = {'cor_type': self.cor, 'use_errors': 2, 'polaxis': self.polaxis, 'output_dim': self.output_dim, \
                           'use_dm1': 1, 'dm1_v': dm1v, 'use_dm2': 1, 'dm2_v': dm2v, 'use_fpm': 1, 'use_lyot_stop': 1,
                           'use_field_stop': 1}

        optics = instrument.CorgiOptics(self._mode, self.bandpass, optics_keywords=optics_keywords, if_quiet=True)

        sim_scene = optics.get_host_star_psf(self.base_scene)
        image_star_corgi = sim_scene.host_star_image.data

        # emccd parameters for excam 
        emccd_dict = {'em_gain': gain, 'cr_rate': 0}  # cr_rate always = 0 

        detector = instrument.CorgiDetector(emccd_dict)

        if self.is_noise_free:
            frame = sim_scene.host_star_image.data 
        else: 
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            image_tot_corgi_sub = sim_scene.image_on_detector.data
            frame = image_tot_corgi_sub

        return frame

    def gitlframe_cgihowfsc(self, cfg, dmlist, peakflux, fixedbp, exptime, crop, lind, cleanrow=1024, cleancol=1024, wfe=None):

        frame = sim_gitlframe(cfg, dmlist, fixedbp, peakflux, exptime, crop, lind, cleanrow=1024, cleancol=1024)

        return frame

    def get_image(self, dm1v, dm2v, exptime, gain, crop, lind, peakflux=1,cleanrow=1024, cleancol=1024, fixedbp=np.zeros((1024, 1024), dtype=bool), wfe=None):
        """
        Get a simulated GITL frame using either corgisim or cgi-howfsc repo's optical model

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

         exptime: Exposure time used when collecting the data in in.      Should be a real scalar > 0 when noise is included. 
         gain: EM gain setting for the EMCCD.  Real scalar >= 1.

        wfe is a placeholder argument for now to keep the option of passing additional wavefront error (e.g. zernike coefficients) to modify the frame generation

        This get_image method should be compatible with both cgi-howfsc and corgisim 
        """
        if self.name == 'corgihowfsc':
            frame = self.gitlframe_corgisim(dm1v, dm2v, fixedbp, exptime, crop, lind, gain, cleanrow, cleancol, wfe)

        elif self.name == 'cgi-howfsc':
            dmlist = [dm1v, dm2v] 
            frame = self.gitlframe_cgihowfsc(self.cfg, dmlist, peakflux, self.cstrat.fixedbp, exptime, crop, lind, cleanrow, cleancol, wfe)

        return frame