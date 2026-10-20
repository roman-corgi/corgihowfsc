# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Load up a coronagraph mode for interactive use with the command line
"""
import os

from astropy.io import fits
import numpy as np

from cal.util import check
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml
import cal.gainmap.gainmap as gm
import cal.gainmap.gm_util as gmu

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')
DATA_PATH = os.path.join(MODEL_PATH, 'homf_dmreg')


def write_seed_from_tvac():

    flatMapFlightDM1 = fits.getdata(os.path.join(OUT_PATH, 'pupil', '1b', 'flatmap_band1_dm1_v.fits'))
    flatMapFlightDM2 = fits.getdata(os.path.join(OUT_PATH, 'pupil', '1b', 'flatmap_band1_dm2_v.fits'))

    fn_dmreg = os.path.join(DATA_PATH, 'howfsc_optical_model_dmreg_only_band_1b.yaml')
    voltageDictDM1 = loadyaml(fn_dmreg)['dms']['DM1']['voltages']
    voltageDictDM2 = loadyaml(fn_dmreg)['dms']['DM2']['voltages']

    fnRefCubeDM1 = os.path.join(DM1_PATH, 'height_cube_integrated.fits')
    fnRefCubeDM2 = os.path.join(DM2_PATH, 'height_cube_integrated.fits')

    fnRefCommandVecDM1 = os.path.join(DM1_PATH, 'height_cube_command_vec.fits')
    fnRefCommandVecDM2 = os.path.join(DM2_PATH, 'height_cube_command_vec.fits')

    fn_xtalk_dm1 = None
    fn_xtalk_dm2 = os.path.join(DM2_PATH, 'crosstalk_dm2_20240719.yaml')
    
    commandMapBeforeDM1 = fits.getdata(os.path.join(DM1_PATH, 'dm1_tvac_flat_dm2adjusted_2024_07_19.fits'))
    commandMapBeforeDM2 = fits.getdata(os.path.join(DM2_PATH, 'dm2_tvac_flat_dm2adjusted_2024_07_19.fits'))
    
    commandMapAfterDM1 = fits.getdata(os.path.join(DM1_PATH, 'dm1_tvac_hlc_dm2adjusted_2024_07_19.fits'))
    commandMapAfterDM2 = fits.getdata(os.path.join(DM2_PATH, 'dm2_tvac_hlc_dm2adjusted_2024_07_19.fits'))

    
    deltaHeightMapDM1 = gmu.compute_delta_height_map_from_command_maps(
        commandMapBeforeDM1,
        commandMapAfterDM1,
        fnRefCubeDM1,
        fnRefCommandVecDM1,
        crosstalk_fn=fn_xtalk_dm1,
    )
    
    deltaHeightMapDM2 = gmu.compute_delta_height_map_from_command_maps(
        commandMapBeforeDM2,
        commandMapAfterDM2,
        fnRefCubeDM2,
        fnRefCommandVecDM2,
        crosstalk_fn=fn_xtalk_dm2,
    )

    v_seed_dm1 = gm.compute_starting_commands_for_flight(
        flatMapFlightDM1,
        deltaHeightMapDM1,
        fnRefCubeDM1,
        fnRefCommandVecDM1,
        voltageDictDM1,
        data_path=DATA_PATH,
    )

    v_seed_dm2 = gm.compute_starting_commands_for_flight(
        flatMapFlightDM2,
        deltaHeightMapDM2,
        fnRefCubeDM2,
        fnRefCommandVecDM2,
        voltageDictDM2,
        data_path=DATA_PATH,
    )

    fn_out_dm1 = os.path.join(MODEL_PATH, 'nfov_band1', 'nfov_band1_360deg', 'hlc_seed_from_tvac_dm1.fits')
    fn_out_dm2 = os.path.join(MODEL_PATH, 'nfov_band1', 'nfov_band1_360deg', 'hlc_seed_from_tvac_dm2.fits')
    fits.writeto(fn_out_dm1, v_seed_dm1, overwrite=True)
    fits.writeto(fn_out_dm2, v_seed_dm2, overwrite=True)

    return v_seed_dm1, v_seed_dm2
    

if __name__ == "__main__":

    write_seed_from_tvac()
