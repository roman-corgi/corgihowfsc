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
from cgisim import rcgisim
import time


class GitlField:
    # Gitl Image class should be able to handle either corgisim, or calling cgisim, following the defintion from sim_gitlframe in howfsc.util.gitlframes
    def __init__(self, name, cfg=None, cstrat=None, Vmag=2.56, sptype='A5V', ref_flag=False, is_noise_free=True,
                 output_dim=51):

        # common configs
        self._mode = 'excam_efield'  # default and the only mode we are using for the camera

        if name not in ['corgihowfsc', 'cgi-howfsc']:
            raise ValueError("name must be 'corgihowfsc' or 'cgi-howfsc'")
        self.name = name

        host_star_properties = {'Vmag': Vmag,
                                'spectral_type': sptype,
                                'magtype': 'vegamag',
                                'ref_flag': ref_flag}
        # cgi-howfsc configs
        self.cfg = cfg
        self.cstrat = cstrat

        # corgisim configs
        self.is_noise_free = is_noise_free  # corgisim only
        self.output_dim = output_dim

        point_source_info = []  # default is just none, tbc whether there should be point source or not
        self.base_scene = scene.Scene(host_star_properties, point_source_info)

    def check_gitl_efield_inputs(self, dm1v, dm2v, fixedbp, exptime, crop, cleanrow, cleancol):
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
            #check.real_positive_scalar(peakflux, 'peakflux', TypeError)

            #check.nonnegative_scalar_integer(lind, 'lind', TypeError)

            #if lind >= len(cfg.sl_list):
            #    raise ValueError('lind must be < len(cfg.sl_list)')

            check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
            check.positive_scalar_integer(cleancol, 'cleancol', TypeError)

    def gitl_efield_cgihowfsc(self, exptime, gain,
                              dm1v, dm2v,
                              cor, bandpass,
                              crop,
                              polaxis,
                              cleanrow, cleancol,
                              fixedbp,
                              wfe=None):
        # Not implemented yet
        real_field, imaginary_field = None, None
        return real_field, imaginary_field

    def gitl_efield_corgi_howfsc(self,
                                 dm1v, dm2v,
                                 cor, bandpass,
                                 crop,
                                 polaxis,
                                 cleanrow, cleancol, wfe=None):

        #self.check_gitl_efield_inputs(dm1v, dm2v, fixedbp, exptime=exptime, crop=crop, cleanrow=cleanrow,
        #                           cleancol=cleancol)

        optics_keywords = {'use_dm1': 1, 'dm1_v': dm1v, 'use_dm2': 1, 'dm2_v': dm2v, 'use_fpm': 1, 'use_lyot_stop': 1,
                           'use_field_stop': 1}

        #optics = instrument.CorgiOptics(self._mode, bandpass, optics_keywords=optic_keywords, if_quiet=True)
        #sim_scene = optics.get_host_star_psf(self.base_scene)

        efield = rcgisim(cgi_mode=self._mode, cor_type=cor, bandpass=bandpass, polaxis=polaxis, param_struct=optics_keywords)

        return efield


    def get_efield(self, dm1v, dm2v,
                   cor, bandpass,
                   crop=None,
                   polaxis=10,
                   wfe=None):
        """
        Get a simulated GITL efield using either corgisim or cgi-howfsc repo's optical model

        wfe is a placeholder argument for now to keep the option of passing additional wavefront error (e.g. zernike coefficients) to modify the frame generation

        This get_image method should be compatible with both cgi-howfsc and corgisim
        """
        if self.name == 'cgi-howfsc':
            print("Not yet implemented")
            return 0

        elif self.name == 'corgihowfsc':
            dmlist = [dm1v, dm2v]
            efields = self.gitl_efield_corgi_howfsc(dmlist[0], dmlist[1], cor, bandpass,
                                                                    crop, polaxis, cleanrow=1024, cleancol=1024, wfe=wfe)

        return efields