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
    def __init__(self, Vmag=2.56, sptype='A5V', ref_flag=False, is_noise_free=True):
        self._mode = 'excam' 
        host_star_properties = {'Vmag': Vmag,
                                'spectral_type': sptype,
                                'magtype': 'vegamag',
                                'ref_flag': ref_flag}

        # CHECK - if this is corgisim,  then we need to pass keywords
        self.is_noise_free = is_noise_free
        # Construct a list of dictionaries for all companion point sources
        point_source_info = []

        # --- Create the Astrophysical Scene ---
        # This Scene object combines the host star and companion(s)
        self.base_scene = scene.Scene(host_star_properties, point_source_info)

        # self.CCD_DEFAULT -> being passed by the cgisim for getting the image
        # NOTE - I do not know what would be the equivalent of this in the corgisim 

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

    def check4gitlframeinputs()

    

    def gen_corgisim_excam_frame_for_gitl(self,
            exptime, gain,
            dm1v, dm2v,
            cor, bandpass,
            crop,
            param_dict={},
            ccd=None,
            polaxis=10,
            cleanrow=1024, cleancol=1024,
    ):

        """
        """
        # Frozen inputs
        # CHECK - how do we get this? 
        fixedbp = np.zeros((cleanrow, cleancol), dtype=bool)  # TEMPORARY

        # TODO - for all these checks, move them into a function, or a part of the init
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

        check.real_positive_scalar(exptime, 'exptime', TypeError)

        if not isinstance(crop, tuple):
            raise TypeError('crop must be a tuple')
        if len(crop) != 4:
            raise TypeError('crop must be a 4-tuple')
        check.nonnegative_scalar_integer(crop[0], 'crop[0]', TypeError)
        check.nonnegative_scalar_integer(crop[1], 'crop[1]', TypeError)
        check.positive_scalar_integer(crop[2], 'crop[2]', TypeError)
        check.positive_scalar_integer(crop[3], 'crop[3]', TypeError)

        check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
        check.positive_scalar_integer(cleancol, 'cleancol', TypeError)


        # TODO: update output dim to be related to the crop stuff
        # output_dim define the size of the output image
        output_dim = 51

        # TODO: update this to be optics_keywords - newer version of corgisim will call this optics keywords 
        proper_keywords = {'cor_type': cor, 'use_errors': 2, 'polaxis': polaxis, 'output_dim': output_dim, \
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
            sim_scene = detector.generate_detector_image(sim_scene, exptime)
            image_tot_corgi_sub = sim_scene.image_on_detector.data
            frame = image_tot_corgi_sub

        # TODO - check if we need to crop the image here, or later
        return frame

        def get_image(self,wfe):
            """
            Placeholder method to get a frame given a wavefront error map

            wfe is a placeholder argument for now to keep the option of passing additional wavefront error (
            e.g. zernike coefficients) to modify the frame generation

            This get_image method should be compatible with both cgi-howfsc and corgisim 
            """

            return NotImplementedError('This method should be implemented in subclasses')