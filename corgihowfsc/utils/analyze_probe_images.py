"""
Create Gaussian probes that correspond to the original sinc-sinc-sine probes,
which have been adjusted to fit within the SPM and LS openings.

Plots are generated that show the DM surface and the pupil mask openings.

NOTES on data types:
    Refer to https://note.nkmk.me/en/python-numpy-dtype-astype/
    i1 is uint8
    f4 is float32 (f for float, and 4 for 4 bytes)
    < = little-endian (LSB first)
    > = big-endian (MSB first)

Example Calls in a Bash Terminal:
python write_gaussian_probes.py --mode 'nfov_band1' --dark_hole '360deg'
python write_gaussian_probes.py --mode 'nfov_band1' --dark_hole 'half_top'
python write_gaussian_probes.py --mode 'spec_band3' --dark_hole 'both_sides' --write
python write_gaussian_probes.py --mode 'wfov_band4' --dark_hole '360deg' --write

"""
import argparse
import os
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files

from howfsc.model.mode import CoronagraphMode
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.insertinto import insertinto as inin
from howfsc.util.loadyaml import loadyaml
from corgihowfsc.utils.cgi_prop_tools import make_dmrel_probe_gaussian

# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
thisFolder = os.path.basename(HERE)
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
probepath = os.path.join(howfscpath, 'model', 'probes')


# %% Functions
def fft2(arrayIn):
    """Perform an energy-conserving 2-D FFT including fftshift."""
    arrayOut = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayIn))) / np.sqrt(arrayIn.shape[0] * arrayIn.shape[1])

    return arrayOut


def plot_gaussian_probes(mode, dark_hole, ni_desired):
    # Plot Gaussian probes, creating them in the same way like from write_gaussian_probes.py

    if 'nfov' in mode:  # nfov_band1, nfov_band2, nfov_band3, nfov_band4
        if '360' in dark_hole:  # 360-degree dark zone
            sigma = 1.
            deltax_act_list = [13, 13, 12]
            deltay_act_list = [8, 9, 9]
        else:
            raise NotImplementedError('Passed dark hole not implemented')

        Rin = 2.8
        Rout = 209.7
        lod_min = Rin
        lod_max = Rout

    else:
        raise NotImplementedError('Passed mode not implemented')

    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args,
                                                                                                           howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps
    homf_dict = loadyaml(cfgfile)
    dmreg_dm1 = homf_dict['dms']['DM1']['registration']

    # # From FALCO:
    lam = homf_dict['sls'][1]['lam']
    surfMax = 4 * np.pi * lam * np.sqrt(ni_desired)  # [meters]
    print('Max probe height in meters to get NI=%.2e is: %.4g' % (ni_desired, surfMax))

    diam_pupil_pix = homf_dict['sls'][1]['epup']['pdp']
    dact = diam_pupil_pix / dmreg_dm1['ppact_cx']

    fn_amp = os.path.join(modelpath, homf_dict['sls'][1]['epup']['afn'])
    fn_ph = os.path.join(modelpath, homf_dict['sls'][1]['epup']['pfn'])
    fn_lyot = os.path.join(modelpath, homf_dict['sls'][1]['lyot']['afn'])
    fn_spm = os.path.join(modelpath, homf_dict['sls'][1]['sp']['afn'])
    fn_fpm = os.path.join(modelpath, homf_dict['sls'][1]['fpm']['afn'])
    fn_dh = os.path.join(modelpath, homf_dict['sls'][1]['dh'])

    amp = fits.getdata(fn_amp)
    ph = fits.getdata(fn_ph)
    lyot = fits.getdata(fn_lyot)
    spm = fits.getdata(fn_spm)
    fpm = fits.getdata(fn_fpm)
    dh = fits.getdata(fn_dh)

    # Read in tie map
    NACT = homf_dict['dms']['DM1']['registration']['nact']
    tiemap = fits.getdata(os.path.join(modelpath, homf_dict['dms']['DM1']['voltages']['tiefn']))
    usable_act_map = np.zeros((NACT, NACT), dtype=bool)
    usable_act_map[tiemap == 0] = True

    probe_name_list = ['gauss0', 'gauss1', 'gauss2']
    probe_ni_map_list = []

    for index_probe, _ in enumerate(deltax_act_list):
        print('*** Probe %d ***' % index_probe)

        deltax_act = deltax_act_list[index_probe]
        deltay_act = deltay_act_list[index_probe]
        probe_name = probe_name_list[index_probe]

        probe_tuple = make_dmrel_probe_gaussian(
            cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act, ycenter=deltay_act, sigma=sigma,
            target=ni_desired, lod_min=lod_min, lod_max=lod_max,
            ind=1, maxiter=5)

        dpv = probe_tuple[0]
        probe_ni_map = probe_tuple[1]
        dh_mask = probe_tuple[2]
        dm_surf = probe_tuple[3]
        pupil_masks = probe_tuple[4]

        probe_ni_map_list.append(probe_ni_map)

        dpv = usable_act_map * dpv

        plt.figure(2 + 10 * index_probe)
        plt.imshow(probe_ni_map, norm=LogNorm(vmin=ni/10, vmax=ni*1.5), cmap='inferno')
        plt.title('Probe-only Intensity')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()

        plt.figure(1)
        plt.title('dpv')
        plt.imshow(dpv)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()


if __name__ == '__main__':

    mode = 'nfov_band1'
    dark_hole = '360deg'
    ni = 1e-7

    plot_gaussian_probes(mode, dark_hole, ni_desired=ni)
