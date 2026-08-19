"""
Compute the new gain map in m/V for a DM about the provided absolute DM maps.

Save out the gain map as a FITS file to the relevant folder.


Actions:
  1. Load the absolute DM map from its FITS file.
  2. Load the gain datacube for that DM.
  3. Load the gain datacube's command voltage vector for that DM.
  4. Run gainmap.gainmap.compute_gainmap_from_gain_cube() via DORBA.
  5. Save the updated gainmap to the specified filepath.

"""
import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits

import __main__ as main

# From cgi-cal
from cal.gainmap.gainmap import compute_gainmap_from_gain_cube

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')

# some default constants
GAIN_CUBE_DM1 = 'gain_cube.fits'
GAIN_CUBE_DM2 = 'gain_cube.fits'
GAIN_CUBE_COMMAND_VEC_DM1 = 'gain_cube_command_vec.fits'
GAIN_CUBE_COMMAND_VEC_DM2 = 'gain_cube_command_vec.fits'
# CORO_MODE_DEFAULT = 'dmreg'  # dmreg for registration only in cgi-orbit, or the actual coro_mode for cgi-aux
THRESHOLD_DEFAULT = 3e-10  # 1e-10 # meters / volt
NACT = 48


def compute_dm_gain_map(
        which_dm,
        fn_dmabs_map,
        threshold=THRESHOLD_DEFAULT,
        out_path=None,
        out_file=None,
        show_plot=True,
        ):
    """
    Calculate a gain map where the gain for each actuator depends on the nominal voltage of the
    actuator. An input voltage map is required.

    Inputs:
      which_dm : int, 1 or 2
      fn_dmabs_map : str, filename for input DM voltage map, with full path, or relative to current path
                   : numpy.ndarray, 2-d real array shape=(48,48) voltage map
      coro_mode : str, name of just the sub-folder into which to write the DM gain maps.

    Ouputs:
      gain_map: fits file, 2-D array containing the new gain map

    Returns:
      fn_gain_map_out : str, filename of written gain map fits file

    """
    # 1. input DM voltage map can be a filename or a 2-d voltage map.
    #    If filename, load the absolute DM map from its FITS file.
    if type(fn_dmabs_map) is str:
        dmabs_map = fits.getdata(fn_dmabs_map)
    elif type(fn_dmabs_map) is np.ndarray:
        dmabs_map = fn_dmabs_map.copy()
    else:
        raise TypeError('input fn_dmabs_map must be either filename string or numpy array')

    fn_gain_map_out = compute_dm_gain_map_base(
        which_dm,
        dmabs_map,
        # bandpass, #coro_mode=coro_mode,
        threshold=threshold,
        out_path=out_path,
        out_file=out_file,
        show_plot=show_plot,
    )

    return fn_gain_map_out


def compute_dm_gain_map_base(
        which_dm,
        dmabs_map,
        # bandpass, #coro_mode=CORO_MODE_DEFAULT,
        threshold=THRESHOLD_DEFAULT,
        out_path=None,
        out_file=None,
        show_plot=True,
        ):

    # Don't allow out of bounds values
    min_val = 0
    max_val = 100
    dmabs_map[dmabs_map < min_val] = min_val
    dmabs_map[dmabs_map > max_val] = max_val

    # 2. Specify the gain datacube for that DM.
    # 3. Specify the gain datacube's command voltage vector for that DM.
    if which_dm == 1:
        gain_cube_fn = os.path.join(DM1_PATH, GAIN_CUBE_DM1)
        cmd_vec_fn = os.path.join(DM1_PATH, GAIN_CUBE_COMMAND_VEC_DM1)

    elif which_dm == 2:
        gain_cube_fn = os.path.join(DM2_PATH, GAIN_CUBE_DM2)
        cmd_vec_fn = os.path.join(DM2_PATH, GAIN_CUBE_COMMAND_VEC_DM2)

    else:
        raise ValueError('which_dm must be integer 1 or 2')

    # 4. Run cal.gainmap.gainmap.compute_gainmap_from_gain_cube().
    # compute_gainmap_from_gain_cube(commandMap, fnRefCube, fnRefCommandVec)
    # input fnRefCube is gain / DM command (i.e. not height)
    inputs = {
        'commandMap': dmabs_map,
        'fnRefCube': (gain_cube_fn),
        'fnRefCommandVec': (cmd_vec_fn),
        'threshold': threshold,
        'min_gain': threshold * np.ones((NACT, NACT)),
    }
    gain_map = compute_gainmap_from_gain_cube(**inputs)

    # 5. Save the updated gainmap to the specified filepath.
    # If the specified output folder doesn't exist, throw an error
    howfsc_path = os.path.join(GEN_MODEL_PATH, 'out', 'dm')

    if out_path is None:
        out_path = howfsc_path

    if out_file is None:
        out_file = f'gain_map_dm{which_dm}.fits'

    fn_gain_map_out = os.path.join(out_path, out_file)

    print(f'Saving new gain map for DM{which_dm} to: {fn_gain_map_out}')
    fits.writeto(fn_gain_map_out, gain_map, overwrite=True)

    # 6. plot gain map, etc. for sanity checking
    if show_plot:

        fig, ax = plt.subplots(nrows=1, ncols=2)
        axim_0 = ax[0,].imshow(dmabs_map, cmap='jet')
        ax[0,].invert_yaxis()
        ax[0,].set_title('Input DM Map')
        plt.colorbar(axim_0, ax=ax[0,])

        axim_1 = ax[1,].imshow(gain_map, cmap='jet')
        ax[1,].invert_yaxis()
        ax[1,].set_title('Output DM Gain Map')
        plt.colorbar(axim_1, ax=ax[1,])

        block = hasattr(main, '__file__')
        if block:
            print('Close figure to continue...')

        fig.tight_layout()
        plt.show(block=block)

    return fn_gain_map_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="python compute_dm_gain_map.py",
        description="Wrapper for cal.gainmap.gainmap.compute_gainmap_from_gain_cube. Given a DM height map, calculate the appropriate gain map")
    parser.add_argument("which_dm", type=int, help="1 or 2, indicating the DM.")
    parser.add_argument("fn_dmabs_map", type=str,
                        help="Full path and name of FITS file containing input absolute DM map.")
    parser.add_argument("--out_path", type=str, default=None,
                        help="Name of output path excluding the file name")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Name of the output file excluding the path.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help="Gain threshold value. Gains below threshold are replaced with this value. [m/V].")
    parser.add_argument("--no_plot", action="store_true",
                        help="Display plots of input and output maps. Default is to show the plots.")

    args = parser.parse_args()

    retvals = compute_dm_gain_map(
        args.which_dm,
        args.fn_dmabs_map,
        threshold=args.threshold,
        out_path=args.out_path,
        out_file=args.out_file,
        show_plot=(not args.no_plot),
    )
