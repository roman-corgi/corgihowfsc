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
import os

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
from howfsc.util.prop_tools import efield, open_efield

# PATHS
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
probepath = os.path.join(howfscpath, 'model', 'probes')


def plot_gaussian_probes(mode, dark_hole, ni_desired):
    # Plot Gaussian probes, creating them in the same way as from write_gaussian_probes.py

    if 'nfov' in mode:  # nfov_band1, nfov_band2, nfov_band3, nfov_band4
        if '360' in dark_hole:  # 360-degree dark zone
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
    band_indices = [0, 1, 2]

    # Define sigma range
    sigma_values = np.arange(0.5, 2.1, 0.1)

    # Dictionary to store probe ni data for each sigma value
    sigma_probe_ni_data = {}

    # Loop over sigma values
    for sigma in sigma_values:
        print(f'\n*** Processing sigma = {sigma:.1f} ***')

        # Initialize arrays to store results for all probes and wavelengths
        probe_ni_maps = []  # Shape: [n_probes, n_bands]

        # First, create the probes using band 1 (as in original code)
        dpv_list = []
        for index_probe, _ in enumerate(deltax_act_list):
            print('*** Creating Probe %d ***' % index_probe)

            deltax_act = deltax_act_list[index_probe]
            deltay_act = deltay_act_list[index_probe]
            probe_name = probe_name_list[index_probe]

            probe_tuple = make_dmrel_probe_gaussian(
                cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act, ycenter=deltay_act, sigma=sigma,
                target=ni_desired, lod_min=lod_min, lod_max=lod_max,
                ind=1, maxiter=5)

            dpv = probe_tuple[0]
            dpv = usable_act_map * dpv
            dpv_list.append(dpv)

        # Now propagate each probe through all three wavelength bands
        for index_probe, dpv in enumerate(dpv_list):
            print('*** Propagating Probe %d through all wavelength bands ***' % index_probe)

            probe_ni_per_wvln = []

            for band_ind in band_indices:
                print(f'  Band {band_ind}')

                # Calculate unocculted electric field for normalization factor in intensity
                iopen = np.abs(open_efield(cfg, dmlist, band_ind)) ** 2
                ipeak = np.max(iopen)
                eref = efield(cfg, dmlist, band_ind)

                # Create probe tuple for this wavelength band
                probed_efield = efield(cfg, [dmlist[0] - dpv, dmlist[1]], band_ind)
                probe_ni_map = np.abs(probed_efield - eref)**2 / ipeak

                probe_ni_per_wvln.append(probe_ni_map)

            probe_ni_maps.append(probe_ni_per_wvln)

        # Store the probe ni data for this sigma
        sigma_probe_ni_data[sigma] = probe_ni_maps

        # Create combined grid: 3 rows (probes) x 4 columns (1 DPV + 3 bands)
        fig, axes = plt.subplots(len(deltax_act_list), len(band_indices) + 1, figsize=(20, 12))

        if len(deltax_act_list) == 1:
            axes = axes.reshape(1, -1)

        im_ni = None
        im_dpv = None

        for i in range(len(deltax_act_list)):
            # Plot DPV map for this probe (first column - far left)
            im_dpv = axes[i, 0].imshow(dpv_list[i], cmap='viridis')
            axes[i, 0].set_title(f'dpv (Volts)')
            axes[i, 0].invert_yaxis()

            # Add rotated probe label on the left side of DPV boxes
            axes[i, 0].text(-0.15, 0.5, f'Probe {i}', transform=axes[i, 0].transAxes,
                           rotation=90, verticalalignment='center', horizontalalignment='center',
                           fontsize=14, fontweight='bold')

            # Plot normalized intensity maps for each band (columns 1-3)
            for j in range(len(band_indices)):
                im_ni = axes[i, j+1].imshow(probe_ni_maps[i][j],
                                           norm=LogNorm(vmin=ni_desired/10, vmax=ni_desired*1.5),
                                           cmap='inferno')
                axes[i, j+1].set_title(f'Band {band_indices[j]}')
                axes[i, j+1].invert_yaxis()

        # Adjust layout first to make room for colorbars
        plt.subplots_adjust(left=0.08, right=0.85, wspace=0.6, hspace=0.3)

        # Add colorbars with precise positioning and consistent sizing
        if im_ni is not None and im_dpv is not None:
            # For DPV - create colorbar close to the right edge of DPV column
            cbar_ax_dpv = fig.add_axes([0.24, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
            cbar_dpv = fig.colorbar(im_dpv, cax=cbar_ax_dpv, label='DPV (Volts)')

            # For normalized intensity - create colorbar on the right side of intensity plots
            cbar_ax_ni = fig.add_axes([0.87, 0.15, 0.01, 0.7])   # [left, bottom, width, height]
            cbar_ni = fig.colorbar(im_ni, cax=cbar_ax_ni, label='Normalized Intensity')

        fig.suptitle(f'Probe Analysis (σ = {sigma:.1f}): DPV Maps and Normalized Intensity Across Wavelength Bands', fontsize=16)

        plt.show()

    return sigma_probe_ni_data


if __name__ == '__main__':

    mode = 'nfov_band1'
    dark_hole = '360deg'
    ni = 1e-7

    plot_gaussian_probes(mode, dark_hole, ni_desired=ni)
