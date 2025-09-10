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

## import packages
from corgisim import scene
from corgisim import instrument
import matplotlib.pyplot as plt
import numpy as np
import proper
from corgisim import outputs
import time

# Gitl Image class should be able to handle either corgisim, or calling cgisim, following the defintion from 
class GitlImage:
    def __init__(self, name, Vmag=2.56, sptype='A5V', ref_flag=False, is_noise_free=True, output_dim=51):
        self._mode = 'excam' # default and the only mode we are using for the camera 
        
        if name not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("name must be 'corgihowfsc' or 'cgi-howfsc'")

        self.name = name
        
        host_star_properties = {'Vmag': Vmag,
                                'spectral_type': sptype,
                                'magtype': 'vegamag',
                                'ref_flag': ref_flag}

        # Default CCD parameters for EXCAM, setup for only passing to cgisim
        self.CCD_DEFAULT = {
            'full_well_serial': 100000,
            'full_well': 80000,
            'dark_rate': 8e-4,
            'cic_noise': 0.016,
            'read_noise': 140.0,
            'bias': 1000,
            'cr_rate': 0,
            'e_per_dn': 6,
            'nbits': 14,
            'numel_gain_register': 604,
        }
        
        # corgisim configs
        self.is_noise_free = is_noise_free # corgisim only 
        self.output_dim = output_dim

        # CORGISIM configs
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
        # TODO - identify the common config and setup for both implementations 
        # Check inputs
        print('Checking inputs for gitlframe generation ...')
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
            # do corgisim check
            # TODO - verify if we need test unique to corgisim
            pass 

        elif self.name == 'cgi-howfsc':
            print('Using cgi-howfsc input checks ...')
            # do cgi-howfsc check 
            # cgi-howfsc version only checks peakflux, but corgisim version checks exptime
            check.real_positive_scalar(exptime, 'exptime', TypeError)

            if not isinstance(crop, tuple):
                raise TypeError('crop must be a tuple')
            if len(crop) != 4:
                raise TypeError('crop must be a 4-tuple')

            check.nonnegative_scalar_integer(crop[0], 'crop[0]', TypeError)
            check.nonnegative_scalar_integer(crop[1], 'crop[1]', TypeError)
            check.positive_scalar_integer(crop[2], 'crop[2]', TypeError)
            check.positive_scalar_integer(crop[3], 'crop[3]', TypeError)

            # FIXME - if it is a noise free case, we do not need to check the peakflux and exptime 
            check.real_positive_scalar(exptime, 'exptime', TypeError)
            check.real_positive_scalar(peakflux, 'peakflux', TypeError)

            check.nonnegative_scalar_integer(lind, 'lind', TypeError)

            if lind >= len(cfg.sl_list):
                raise ValueError('lind must be < len(cfg.sl_list)')
            check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
            check.positive_scalar_integer(cleancol, 'cleancol', TypeError)

    def gitlframe_corgisim(self,
            exptime, gain,
            dm1v, dm2v,
            cor, bandpass,
            crop,
            param_dict,
            ccd,
            polaxis,
            cleanrow, cleancol, fixedbp, wfe=None):

        # # Frozen inputs
        # fixedbp = np.zeros((cleanrow, cleancol), dtype=bool)  # TEMPORARY - 

        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        # TODO: update this to be optics_keywords - newer version of corgisim will call this optics keywords 
        proper_keywords = {'cor_type': cor, 'use_errors': 2, 'polaxis': polaxis, 'output_dim': self.output_dim, \
                           'use_dm1': 1, 'dm1_v': dm1v, 'use_dm2': 1, 'dm2_v': dm2v, 'use_fpm': 1, 'use_lyot_stop': 1,
                           'use_field_stop': 1}

        optics = instrument.CorgiOptics(self._mode, bandpass, proper_keywords=proper_keywords, if_quiet=True)

        sim_scene = optics.get_host_star_psf(self.base_scene)
        image_star_corgi = sim_scene.host_star_image.data

        # emccd parameters for excam
        emccd_keywords = {'em_gain': gain, 'cr_rate': 0}
        detector = instrument.CorgiDetector(emccd_keywords)

        if self.is_noise_free:
            frame = sim_scene.host_star_image.data 
        else: 
            # TODO - check if we need to crop the image here, or later
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            image_tot_corgi_sub = sim_scene.image_on_detector.data
            frame = image_tot_corgi_sub

        return frame

    def gitlframe_cgihowfsc(self, exptime, gain,
            dm1v, dm2v,
            cor, bandpass,
            crop,
            param_dict,
            ccd,
            polaxis,
            cleanrow, cleancol, 
            fixedbp,
            wfe=None):

        # Frozen inputs
        # TEMPORARY. For cgi-howfsc, it should be imported from cstrat (a class containing variables for control strategy)
        self.check_gitlframeinputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow, cleancol=cleancol)

        # FIXME - maek this work
        # sim_gitlframe(cfg, dmlist, fixedbp, peakflux, exptime, crop, lind,
        #                 cleanrow=1024, cleancol=1024)
        # TODO - call sim_gitlframe

        pass

    def get_image(self, exptime, gain,
            dm1v, dm2v,
            cor, bandpass,
            crop=None,
            param_dict={},
            ccd=None,
            polaxis=10,
            cleanrow=1024, cleancol=1024, 
            fixedbp=np.zeros((1024, 1024), dtype=bool),
            wfe=None):
        """
        Placeholder method to get a frame given a wavefront error map

        wfe is a placeholder argument for now to keep the option of passing additional wavefront error (
        e.g. zernike coefficients) to modify the frame generation

        This get_image method should be compatible with both cgi-howfsc and corgisim 
        """
        if self.name == 'corgihowfsc':
            self.gitlframe_corgisim(exptime, gain,
                dm1v, dm2v,
                cor, bandpass,
                crop=crop,
                param_dict=param_dict,
                ccd=ccd,
                polaxis=polaxis,
                cleanrow=cleanrow, cleancol=cleancol,
                fixedbp=fixedbp,
                wfe=wfe)

        elif self.name == 'cgi-howfsc':
            self.gitlframe_cgihowfsc(exptime, gain,
                dm1v, dm2v,
                cor, bandpass,
                crop=crop,
                param_dict=param_dict,
                ccd=ccd,
                polaxis=polaxis,
                cleanrow=cleanrow, cleancol=cleancol,
                fixedbp=fixedbp,
                wfe=wfe)