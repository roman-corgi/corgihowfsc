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
import matplotlib.animation as animation
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
from howfsc.util.prop_tools import make_dmrel_probe

# PATHS
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
probepath = os.path.join(howfscpath, 'model', 'probes')


def extract_annulus_radii_pixels(mask, center_row, center_col):
    """
    Extract the inner and outer radii of an annular mask in pixels.

    Parameters:
    -----------
    mask : np.ndarray
        Boolean annular mask
    center_row, center_col : float
        Center coordinates of the annulus

    Returns:
    --------
    r_inner, r_outer : float
        Inner and outer radii in pixels
    """
    # Create coordinate arrays
    rows, cols = np.ogrid[:mask.shape[0], :mask.shape[1]]
    r = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)

    # Find radii where mask transitions from False to True (inner edge) and True to False (outer edge)
    mask_radii = r[mask]

    if len(mask_radii) == 0:
        return 0, 0

    r_inner = np.min(mask_radii)
    r_outer = np.max(mask_radii)

    return r_inner, r_outer


def draw_circle_on_axis(ax, center_row, center_col, radius, **kwargs):
    """
    Draw a circle on a matplotlib axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on
    center_row, center_col : float
        Center coordinates
    radius : float
        Radius in pixels
    **kwargs : dict
        Additional arguments passed to matplotlib Circle
    """
    from matplotlib.patches import Circle

    # Note: matplotlib uses (x, y) = (col, row) convention
    circle = Circle((center_col, center_row), radius, fill=False, **kwargs)
    ax.add_patch(circle)


def plot_gaussian_probes(mode, dark_hole, ni_desired, output_path=None, show_dm_surface=False):
    """
    Plot Gaussian probes, creating them in the same way as from write_gaussian_probes.py

    Parameters:
    -----------
    mode : str
        Coronagraph mode (e.g., 'nfov_band1')
    dark_hole : str
        Dark hole specification (e.g., '360deg')
    ni_desired : float
        Desired normalized intensity
    output_path : str, optional
        Path to save PDF plots and animation. If None, plots are displayed interactively.
    show_dm_surface : bool, optional
        If True, show DM surface overlays with pupil masks in the first column instead of DPV maps.
        Default is False (show DPV maps).

        The DM surface overlay is calculated as:
        overlay = 3 * dm_surf / np.max(dm_surf) + pupil_masks / np.max(pupil_masks)
    """
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

    # Read in tie map
    NACT = homf_dict['dms']['DM1']['registration']['nact']
    tiemap = fits.getdata(os.path.join(modelpath, homf_dict['dms']['DM1']['voltages']['tiefn']))
    usable_act_map = np.zeros((NACT, NACT), dtype=bool)
    usable_act_map[tiemap == 0] = True

    probe_name_list = ['gauss0', 'gauss1', 'gauss2']
    band_indices = [0, 1, 2]

    # Define sigma range
    sigma_values = np.arange(0.1, 2.1, 0.1)

    # Dictionary to store probe ni data for each sigma value
    sigma_probe_ni_data = {}

    # List to store figures for animation
    animation_figures = []

    # Get the dark hole mask to extract radii, and pupil masks from a temporary probe
    temp_probe_tuple = make_dmrel_probe_gaussian(
        cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act_list[0], ycenter=deltay_act_list[0], sigma=1.0,
        target=ni_desired, lod_min=lod_min, lod_max=lod_max,
        ind=1, maxiter=1)
    dh_mask = temp_probe_tuple[2]
    pupil_masks = temp_probe_tuple[4]

    # Extract radii from dh_mask for circle overlays
    dh_mask_center_row = dh_mask.shape[0] // 2
    dh_mask_center_col = dh_mask.shape[1] // 2
    r_inner_full, r_outer_full = extract_annulus_radii_pixels(dh_mask, dh_mask_center_row, dh_mask_center_col)
    print(f'Dark hole mask radii: inner = {r_inner_full:.1f} pixels, outer = {r_outer_full:.1f} pixels')

    # Loop over sigma values
    for sigma in sigma_values:
        print(f'\n*** Processing sigma = {sigma:.1f} ***')

        # Initialize arrays to store results for all probes and wavelengths
        probe_ni_maps = []  # Shape: [n_probes, n_bands]

        # First, create the probes using band 1 (as in original code)
        dpv_list = []
        dm_surfaces = []
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

            dm_surf = probe_tuple[3]
            dm_surfaces.append(dm_surf)

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

        # Convert radii to cropped coordinates
        # The crop shifts the center, so the center in cropped coordinates is at the crop_radius
        crop_center_row = crop_radius
        crop_center_col = crop_radius

        # The radii should be the same scale since we're just cropping, not rescaling
        r_inner_crop = r_inner_full
        r_outer_crop = r_outer_full

        for i in range(len(deltax_act_list)):
            # Plot DPV map or DM surface overlay for this probe (first column - far left)
            if show_dm_surface:
                # Create overlay: 3 * dm_surf / np.max(dm_surf) + pupil_masks / np.max(pupil_masks)
                overlay = 3 * dm_surfaces[i] / np.max(dm_surfaces[i]) + pupil_masks / np.max(pupil_masks)
                im_dpv = axes[i][0].imshow(overlay, cmap='Greys_r')
                axes[i][0].set_title(f'DM Surface + Pupil')
            else:
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

                # Add white dashed circles for dark hole mask boundaries
                draw_circle_on_axis(axes[i][j+1], crop_center_row, crop_center_col,
                                   r_inner_crop, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
                draw_circle_on_axis(axes[i][j+1], crop_center_row, crop_center_col,
                                   r_outer_crop, color='white', linestyle='--', linewidth=1.5, alpha=0.8)

        # Adjust layout - no longer needed since we manually positioned everything
        # plt.subplots_adjust(left=0.08, right=0.85, wspace=0.6, hspace=0.3)

        # Add colorbars with precise positioning and consistent sizing
        if im_ni is not None and im_dpv is not None:
            # For DPV - create colorbar close to the right edge of DPV column
            cbar_ax_dpv = fig.add_axes([0.24, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
            cbar_dpv = fig.colorbar(im_dpv, cax=cbar_ax_dpv)
            if show_dm_surface:
                cbar_ax_dpv.set_title('nm', fontsize=16, pad=10)
            else:
                cbar_ax_dpv.set_title('Volts', fontsize=16, pad=10)
            cbar_ax_dpv.tick_params(labelsize=20)  # Double the default tick label size

            # For normalized intensity - create colorbar on the right side of intensity plots
            cbar_ax_ni = fig.add_axes([0.87, 0.15, 0.01, 0.7])   # [left, bottom, width, height]
            cbar_ni = fig.colorbar(im_ni, cax=cbar_ax_ni)
            cbar_ni.set_label('Normalized Intensity', fontsize=16)
            cbar_ax_ni.tick_params(labelsize=20)  # Double the default tick label size

        fig.suptitle(f'Gaussian probes (σ = {sigma:.1f}): Normalized intensity across imaging bands', fontsize=20)

        # Save individual plot as PDF if output path is provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            pdf_filename = os.path.join(output_path, f'gaussian_probes_sigma_{sigma:.1f}.pdf')
            fig.savefig(pdf_filename, format='pdf', dpi=300)
            print(f'Saved PDF: {pdf_filename}')

        # Store figure for animation
        animation_figures.append(fig)

        # Close figure to save memory if not displaying
        if output_path:
            plt.close(fig)
        else:
            plt.show()

    # Create MP4 animation if output path is provided
    if output_path and animation_figures:
        print('\nCreating animation...')

        # Create a new figure for animation
        anim_fig = plt.figure(figsize=(20, 12))

        # Use a simpler approach: save frames as temporary images and create video
        temp_frames = []
        for i, fig in enumerate(animation_figures):
            temp_filename = os.path.join(output_path, f'temp_frame_{i:03d}.png')
            fig.savefig(temp_filename, format='png', dpi=150)
            temp_frames.append(temp_filename)

        # Try to create MP4 using matplotlib animation
        mp4_filename = os.path.join(output_path, 'gaussian_probes_sigma_sweep.mp4')
        gif_filename = os.path.join(output_path, 'gaussian_probes_sigma_sweep.gif')

        animation_created = False

        # Try FFmpeg for MP4 first
        if 'ffmpeg' in animation.writers.list():
            try:
                # Create animation by reading back the saved frames
                def animate_frame(frame_num):
                    plt.clf()
                    img = plt.imread(temp_frames[frame_num])
                    plt.imshow(img, origin='upper')  # Fix orientation
                    plt.axis('off')
                    plt.tight_layout()

                anim = animation.FuncAnimation(anim_fig, animate_frame, frames=len(temp_frames),
                                             interval=2000, repeat=True)

                # Save animation with 2 seconds per frame (0.5 fps)
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=0.5, metadata=dict(artist='CORGIHOWFSC'), bitrate=1800)
                anim.save(mp4_filename, writer=writer)

                print(f'Saved MP4 animation: {mp4_filename}')
                animation_created = True

            except Exception as e:
                print(f'Could not create MP4 animation: {e}')

        # Try Pillow for GIF as fallback
        if not animation_created and 'pillow' in animation.writers.list():
            try:
                # Create animation by reading back the saved frames
                def animate_frame(frame_num):
                    plt.clf()
                    img = plt.imread(temp_frames[frame_num])
                    plt.imshow(img, origin='upper')  # Fix orientation
                    plt.axis('off')
                    plt.tight_layout()

                anim = animation.FuncAnimation(anim_fig, animate_frame, frames=len(temp_frames),
                                             interval=2000, repeat=True)

                # Save animation as GIF with 2 seconds per frame
                Writer = animation.writers['pillow']
                writer = Writer(fps=0.5, metadata=dict(artist='CORGIHOWFSC'))
                anim.save(gif_filename, writer=writer)

                print(f'Saved GIF animation: {gif_filename}')
                animation_created = True

            except Exception as e:
                print(f'Could not create GIF animation: {e}')

        if not animation_created:
            print('Could not create animation file.')
            print(f'Individual frames have been saved as PNG files in: {output_path}')
            print('Frame filenames: temp_frame_000.png, temp_frame_001.png, etc.')
            # Keep the temporary frame files if no animation was created
            temp_frames = []

        # Clean up temporary frame files (only if animation was successfully created)
        for temp_file in temp_frames:
            try:
                os.remove(temp_file)
            except:
                pass

        plt.close(anim_fig)


def plot_sinc_probes(dpv_list_sincs, mode, dark_hole, ni_desired, output_path=None):
    """
    Plot sinc-based probes in a grid similar to Gaussian probes, but only showing DPV maps.

    Parameters:
    -----------
    dpv_list_sincs : list
        List of 3 sinc probe DPV arrays (DM voltage maps)
    mode : str
        Coronagraph mode (e.g., 'nfov_band1')
    dark_hole : str
        Dark hole specification (e.g., '360deg')
    ni_desired : float
        Desired normalized intensity for scaling NI plots
    output_path : str, optional
        Path to save PDF plots. If None, plots are displayed interactively.

    Returns:
    --------
    probe_ni_data : dict
        Dictionary containing probe normalized intensity data for all wavelength bands
    """
    if len(dpv_list_sincs) != 3:
        raise ValueError("dpv_list_sincs must contain exactly 3 sinc probes")

    if 'nfov' in mode:  # nfov_band1, nfov_band2, nfov_band3, nfov_band4
        if '360' in dark_hole:  # 360-degree dark zone
            rot_list = [0, 0, 90]
            phase_list = 90 + np.array([0, 60, 120])
            deltax_act = 13
            deltay_act = 8
        else:
            raise NotImplementedError('Passed dark hole not implemented')

        Rin = 2.8
        Rout = 209.7
        ximin = 0
        ximax = Rout+1
        etamax = ximax
        etamin = -etamax
        lod_min = Rin
        lod_max = Rout

    else:
        raise NotImplementedError('Passed mode not implemented')

    args = get_args(mode=mode, dark_hole=dark_hole)
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)

    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps
    homf_dict = loadyaml(cfgfile)

    dmreg_dm1 = homf_dict['dms']['DM1']['registration']
    diam_pupil_pix = homf_dict['sls'][1]['epup']['pdp']
    dact = diam_pupil_pix/dmreg_dm1['ppact_cx']

    # Get the dark hole mask to extract radii, and pupil masks from a temporary probe
    temp_probe_tuple = make_dmrel_probe(
        cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act, ycenter=deltay_act, clock=rot_list[0],
        ximin=ximin, ximax=ximax,
        etamin=etamin, etamax=etamax,
        phase=phase_list[0], target=ni_desired, lod_min=lod_min, lod_max=lod_max,
        ind=1, maxiter=5)
    dh_mask = temp_probe_tuple[2]
    pupil_masks = temp_probe_tuple[4]

    probe_name_list = ['sinc0', 'sinc1', 'sinc2']
    band_indices = [0, 1, 2]

    # Extract radii from dh_mask for circle overlays
    dh_mask_center_row = dh_mask.shape[0] // 2
    dh_mask_center_col = dh_mask.shape[1] // 2
    r_inner_full, r_outer_full = extract_annulus_radii_pixels(dh_mask, dh_mask_center_row, dh_mask_center_col)
    print(f'Dark hole mask radii: inner = {r_inner_full:.1f} pixels, outer = {r_outer_full:.1f} pixels')

    # Propagate sinc probes through all wavelength bands
    probe_ni_maps = []  # Shape: [n_probes, n_bands]

    for index_probe, dpv in enumerate(dpv_list_sincs):
        print(f'*** Propagating Sinc Probe {index_probe} through all wavelength bands ***')

        probe_ni_per_wvln = []
        dm_surfaces = []

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

            dind = 0  # Only using DM1 to probe
            nrow, ncol = dh_mask.shape
            dm_surf = dmhtoph(
                nrow=nrow,
                ncol=ncol,
                dmin=dpv,
                nact=cfg.dmlist[dind].registration['nact'],
                inf_func=cfg.dmlist[dind].registration['inf_func'],
                ppact_d=cfg.dmlist[dind].registration['ppact_d'],
                ppact_cx=cfg.dmlist[dind].registration['ppact_cx'],
                ppact_cy=cfg.dmlist[dind].registration['ppact_cy'],
                dx=cfg.dmlist[dind].registration['dx'],
                dy=cfg.dmlist[dind].registration['dy'],
                thact=cfg.dmlist[dind].registration['thact'],
                flipx=cfg.dmlist[dind].registration['flipx'],
            )

        probe_ni_maps.append(probe_ni_per_wvln)
        dm_surfaces.append(dm_surf)

    # Create figure with manual subplot positioning (similar to Gaussian probes)
    fig = plt.figure(figsize=(20, 12))

    # Define subplot positions manually with better spacing
    subplot_positions = {
        'dpv': [0.05, 0.22],  # x_start, x_end for DPV column
        'ni': [0.32, 0.85]    # x_start, x_end for NI columns
    }

    row_height = 0.22
    row_starts = [0.68, 0.42, 0.16]  # y positions for 3 rows

    axes = []
    for i in range(len(dpv_list_sincs)):
        row_axes = []

        # DPV subplot - centered in the DPV area
        dpv_ax = fig.add_subplot(len(dpv_list_sincs), 4, i*4 + 1)
        dpv_width = (subplot_positions['dpv'][1] - subplot_positions['dpv'][0]) * 0.8
        dpv_x_offset = (subplot_positions['dpv'][1] - subplot_positions['dpv'][0] - dpv_width) / 2
        dpv_pos = [subplot_positions['dpv'][0] + dpv_x_offset, row_starts[i], dpv_width, row_height]
        dpv_ax.set_position(dpv_pos)
        row_axes.append(dpv_ax)

        # NI subplots (3 columns) with gaps between them
        ni_total_width = subplot_positions['ni'][1] - subplot_positions['ni'][0]
        ni_width = ni_total_width / 3.5
        ni_gap = ni_width * 0.25

        for j in range(3):
            ni_ax = fig.add_subplot(len(dpv_list_sincs), 4, i*4 + j + 2)
            ni_x = subplot_positions['ni'][0] + j * (ni_width + ni_gap)
            ni_pos = [ni_x, row_starts[i], ni_width, row_height]
            ni_ax.set_position(ni_pos)
            row_axes.append(ni_ax)

        axes.append(row_axes)

    im_ni = None
    im_dpv = None

    # Calculate dynamic cropping boundaries for NI images
    sample_image_shape = probe_ni_maps[0][0].shape
    center_row = sample_image_shape[0] // 2
    center_col = sample_image_shape[1] // 2
    crop_radius = 26

    crop_min_row = center_row - crop_radius
    crop_max_row = center_row + crop_radius
    crop_min_col = center_col - crop_radius
    crop_max_col = center_col + crop_radius

    # Convert radii to cropped coordinates
    crop_center_row = crop_radius
    crop_center_col = crop_radius

    # Scale the radii to fit within the cropped image - these are likely too large
    # The crop is only 52x52 pixels, but the original radii might be much larger
    # We need to keep the relative scaling but ensure circles are visible
    if r_outer_full > crop_radius:
        # If outer radius is larger than crop radius, scale both radii down
        scale_factor = (crop_radius * 0.9) / r_outer_full  # Scale to 90% of crop radius
        r_inner_crop = r_inner_full * scale_factor
        r_outer_crop = r_outer_full * scale_factor
        print(f'Scaled circle radii for visibility: inner = {r_inner_crop:.1f}, outer = {r_outer_crop:.1f} (crop pixels)')
    else:
        # If circles fit within crop, use original radii
        r_inner_crop = r_inner_full
        r_outer_crop = r_outer_full

    # Plot the probes
    for i in range(len(dpv_list_sincs)):
        # Plot DPV map for this probe (first column)
        im_dpv = axes[i][0].imshow(dpv_list_sincs[i], cmap='Greys_r')
        axes[i][0].set_title(f'Sinc probe (Volts)')
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

            # Add white dashed circles for dark hole mask boundaries
            if r_inner_crop > 0 and r_outer_crop > 0:
                draw_circle_on_axis(axes[i][j+1], crop_center_row, crop_center_col,
                                   r_inner_crop, color='white', linestyle='--', linewidth=2.0, alpha=0.9)
                draw_circle_on_axis(axes[i][j+1], crop_center_row, crop_center_col,
                                   r_outer_crop, color='white', linestyle='--', linewidth=2.0, alpha=0.9)

                # Add a small center dot for reference (optional debug)
                if j == 0 and i == 0:  # Only on first plot to avoid clutter
                    axes[i][j+1].plot(crop_center_col, crop_center_row, 'w+', markersize=8, markeredgewidth=2)
                    print(f'Added DH circles to sinc NI plots: inner={r_inner_crop:.1f}, outer={r_outer_crop:.1f} pixels')
            else:
                print('Warning: Circle radii are zero or negative, skipping circle overlay')

    # Add colorbars with precise positioning and consistent sizing
    if im_ni is not None and im_dpv is not None:
        # For DPV - create colorbar close to the right edge of DPV column
        cbar_ax_dpv = fig.add_axes([0.24, 0.15, 0.01, 0.7])
        cbar_dpv = fig.colorbar(im_dpv, cax=cbar_ax_dpv)
        cbar_ax_dpv.set_title('Volts', fontsize=16, pad=10)
        cbar_ax_dpv.tick_params(labelsize=20)

        # For normalized intensity - create colorbar on the right side of intensity plots
        cbar_ax_ni = fig.add_axes([0.87, 0.15, 0.01, 0.7])
        cbar_ni = fig.colorbar(im_ni, cax=cbar_ax_ni)
        cbar_ni.set_label('Normalized Intensity', fontsize=16)
        cbar_ax_ni.tick_params(labelsize=20)

    fig.suptitle(f'Sinc probes: Normalized intensity across imaging bands', fontsize=20)

    # Save or show plot
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        pdf_filename = os.path.join(output_path, 'sinc_probes.pdf')
        fig.savefig(pdf_filename, format='pdf', dpi=300)
        print(f'Saved PDF: {pdf_filename}')
        plt.close(fig)
    else:
        plt.show()

    # Prepare return data
    probe_ni_data = {
        'probe_ni_maps': probe_ni_maps,
        'dpv_list': dpv_list_sincs,
        'band_indices': band_indices,
        'probe_names': probe_name_list
    }

    return probe_ni_data


if __name__ == '__main__':

    analysis_path = '/Users/ilaginja/Nextcloud/Areas/RomanCPP/alternate_probes/probe_comparison/active_analysis/sigma_sweep_1e-7_analysis'
    mode = 'nfov_band1'
    dark_hole = '360deg'
    ni = 1e-7

    ### Gaussian probes

    # Example 1: Show DPV maps for Gaussian probes
    # print("Creating plots with DPV maps...")
    # plot_gaussian_probes(mode, dark_hole, ni_desired=ni, output_path=analysis_path, show_dm_surface=False)

    # Example 2: Show DM surface overlays with pupil masks for Gaussian probes
    # print("Creating plots with DM surface overlays...")
    # plot_gaussian_probes(mode, dark_hole, ni_desired=ni, output_path=analysis_path, show_dm_surface=True)

    ### Default probes

    dpv_list_sincs = []   # Baseline sinc-sinc-sine probes, originally scaled to 1e-5
    scale = 0.13
    cos = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_cos.fits') * scale
    dpv_list_sincs.append(cos)
    sinlr = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinlr.fits') * scale
    dpv_list_sincs.append(sinlr)
    sinud = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinud.fits') * scale
    dpv_list_sincs.append(sinud)

    plot_sinc_probes(dpv_list_sincs, mode, dark_hole, ni_desired=ni, output_path=analysis_path)
