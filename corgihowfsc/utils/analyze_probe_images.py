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

        # Create figure with manual subplot positioning
        fig = plt.figure(figsize=(20, 12))

        # Define subplot positions manually with better spacing
        # DPV column: wider for better spacing, NI columns: spread out more
        subplot_positions = {
            'dpv': [0.05, 0.22],  # x_start, x_end for DPV column (wider)
            'ni': [0.32, 0.85]    # x_start, x_end for NI columns (wider span)
        }

        row_height = 0.22  # Slightly smaller to create more vertical gap
        row_starts = [0.68, 0.42, 0.16]  # y positions for 3 rows with more spacing

        axes = []
        for i in range(len(deltax_act_list)):
            row_axes = []

            # DPV subplot - centered in the DPV area
            dpv_ax = fig.add_subplot(len(deltax_act_list), 4, i*4 + 1)
            dpv_width = (subplot_positions['dpv'][1] - subplot_positions['dpv'][0]) * 0.8  # 80% width for spacing
            dpv_x_offset = (subplot_positions['dpv'][1] - subplot_positions['dpv'][0] - dpv_width) / 2
            dpv_pos = [subplot_positions['dpv'][0] + dpv_x_offset, row_starts[i],
                      dpv_width, row_height]
            dpv_ax.set_position(dpv_pos)
            row_axes.append(dpv_ax)

            # NI subplots (3 columns) with gaps between them
            ni_total_width = subplot_positions['ni'][1] - subplot_positions['ni'][0]
            ni_width = ni_total_width / 3.5  # Original width to keep sizes the same
            ni_gap = ni_width * 0.25  # Increased gap between NI plots

            for j in range(3):
                ni_ax = fig.add_subplot(len(deltax_act_list), 4, i*4 + j + 2)
                ni_x = subplot_positions['ni'][0] + j * (ni_width + ni_gap)
                ni_pos = [ni_x, row_starts[i], ni_width, row_height]
                ni_ax.set_position(ni_pos)
                row_axes.append(ni_ax)

            axes.append(row_axes)

        im_ni = None
        im_dpv = None

        # Calculate dynamic cropping boundaries for NI images - 30 pixels from center in each direction
        # Get the image dimensions from the first probe to determine center
        sample_image_shape = probe_ni_maps[0][0].shape
        center_row = sample_image_shape[0] // 2
        center_col = sample_image_shape[1] // 2
        crop_radius = 26

        crop_min_row = center_row - crop_radius
        crop_max_row = center_row + crop_radius
        crop_min_col = center_col - crop_radius
        crop_max_col = center_col + crop_radius

        for i in range(len(deltax_act_list)):
            # Plot DPV map for this probe (first column - far left)
            im_dpv = axes[i][0].imshow(dpv_list[i], cmap='Greys_r')
            axes[i][0].set_title(f'$\Delta$ probe (Volts)')
            axes[i][0].invert_yaxis()

            # Add rotated probe label on the left side of DPV boxes
            axes[i][0].text(-0.15, 0.5, f'Probe {i}', transform=axes[i][0].transAxes,
                           rotation=90, verticalalignment='center', horizontalalignment='center',
                           fontsize=16, fontweight='bold')

            # Plot normalized intensity maps for each band (columns 1-3)
            for j in range(len(band_indices)):
                # Crop the NI image to show only the central square with valid log data
                cropped_ni = probe_ni_maps[i][j][crop_min_row:crop_max_row, crop_min_col:crop_max_col]
                im_ni = axes[i][j+1].imshow(cropped_ni,
                                           norm=LogNorm(vmin=ni_desired/10, vmax=ni_desired*1.5),
                                           cmap='inferno')
                axes[i][j+1].set_title(f'Band {band_indices[j]}')
                axes[i][j+1].invert_yaxis()

        # Adjust layout - no longer needed since we manually positioned everything
        # plt.subplots_adjust(left=0.08, right=0.85, wspace=0.6, hspace=0.3)

        # Add colorbars with precise positioning and consistent sizing
        if im_ni is not None and im_dpv is not None:
            # For DPV - create colorbar close to the right edge of DPV column
            cbar_ax_dpv = fig.add_axes([0.24, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
            cbar_dpv = fig.colorbar(im_dpv, cax=cbar_ax_dpv)
            cbar_ax_dpv.set_title('Volts', fontsize=16, pad=10)
            cbar_ax_dpv.tick_params(labelsize=20)  # Double the default tick label size

            # For normalized intensity - create colorbar on the right side of intensity plots
            cbar_ax_ni = fig.add_axes([0.87, 0.15, 0.01, 0.7])   # [left, bottom, width, height]
            cbar_ni = fig.colorbar(im_ni, cax=cbar_ax_ni)
            cbar_ni.set_label('Normalized Intensity', fontsize=16)
            cbar_ax_ni.tick_params(labelsize=20)  # Double the default tick label size

        fig.suptitle(f'Gaussian probes (σ = {sigma:.1f}): Normalized intensity across imaging bands', fontsize=20)

        plt.show()

    return sigma_probe_ni_data


if __name__ == '__main__':

    mode = 'nfov_band1'
    dark_hole = '360deg'
    ni = 1e-7

    plot_gaussian_probes(mode, dark_hole, ni_desired=ni)
