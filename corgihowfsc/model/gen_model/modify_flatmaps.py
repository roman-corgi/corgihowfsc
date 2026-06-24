# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test accuracy of SPAM calibration."""
import os
import pathlib
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import pandas as pd

from cal.util.constrain_dm import constrain_dm
from cal.util.loadyaml import loadyaml
from cal.util.insertinto import insertinto
from cal.psffit.psffit import psffit

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')
FLATMAP_PATH = os.path.join(IN_PATH, 'flatmaps')
FLATMAP_ORIG_PATH = os.path.join(IN_PATH, 'flatmaps', 'orig')

V_MIN = 20


def modify_flatmaps():
    """Set zero voltages for good actuators to a nonzero value."""
    flagPlot = True

    for dm_num in [1, 2]:
    
        tie_map = fits.getdata(os.path.join(MODEL_PATH, 'dm%d' % dm_num, 'tied_actuator_map.fits'))
        
        for band_num in [1, 2, 3, 4]:
        
            if band_num == 1:
                band = '1b'
            elif band_num == 2:
                band = '2c'
            elif band_num == 3:
                band = '3c'
            elif band_num == 4:
                band = '4b'
            
            fn = 'band%d_flat_wfe_dm%d_v.fits' % (band_num, dm_num)
            v_dm = fits.getdata(os.path.join(FLATMAP_ORIG_PATH, fn))
            
            # Adjust the voltages
            atol = 1 #10*np.finfo(float).eps
            v_dm[v_dm < atol] = V_MIN
            
            flatsurfacemap = np.zeros((48, 48))
            v_dm = constrain_dm(
                v_dm,
                flatsurfacemap,
                tie_map,
            )
            
            fn_out_base = 'flatmap_band%d_dm%d_v.fits' % (band_num, dm_num)
            
            fn_out = os.path.join(FLATMAP_PATH, fn_out_base)
            fits.writeto(fn_out, v_dm, overwrite=True)
            
            fn_out = os.path.join(OUT_PATH, 'pupil', band, fn_out_base)
            fits.writeto(fn_out, v_dm, overwrite=True)
            
            
    return None


if __name__ == '__main__':
    modify_flatmaps()
