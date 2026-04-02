from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check
from howfsc.util.prop_tools import efield, open_efield
from howfsc.util.loadyaml import loadyaml

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.utils.cgi_prop_tools import make_dmrel_probe_gaussian

from astropy.io import fits
import csv
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import json

import os


def analyze_probe_set(cfg, dmlist, dpv_list, dh_mask, ind):
    """
    Analyze a set of probes.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.
     dpv_list: list of 3 DM probe voltages.
     dh_maks: boolean mask for the dark hole region, in the focal plane.
     ind: index of cfg wavelength to use for model

    Returns:
        averages: 2D array (2, 3) - average DH intensities for each probe (positive and negative)
        stddevs: 2D array (2, 3) - standard deviations of DH intensities for each probe
        ptvs: 2D array (2, 3) - peak-to-valley values of DH intensities for each probe
        efields: 4D array (2, 3, *field_shape) - electric fields for each probe
        intensities: 4D array (2, 3, *field_shape) - intensities for each probe
    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    # Calculate unocculted electric field for normalization factor in intensity
    iopen = np.abs(open_efield(cfg, dmlist, ind))**2
    ipeak = np.max(iopen)
    eref = efield(cfg, dmlist, ind)

    # Calculate the electric field for each probe, positive and negative
    efields_pos = []
    intensities_pos = []
    efields_neg = []
    intensities_neg = []
    for i, probe_v in enumerate(dpv_list):
        probed_field_pos = efield(cfg, [dmlist[0] + probe_v, dmlist[1]], ind)
        efields_pos.append((probed_field_pos - eref) / np.sqrt(ipeak))
        intensities_pos.append(np.abs(probed_field_pos - eref)**2 / ipeak)

        probed_field_neg = efield(cfg, [dmlist[0] - probe_v, dmlist[1]], ind)
        efields_neg.append((probed_field_neg - eref) / np.sqrt(ipeak))
        intensities_neg.append(np.abs(probed_field_neg - eref)**2 / ipeak)

    # Measure DH quantities for each probe
    averages_pos = [np.mean(intensity[dh_mask]) for intensity in intensities_pos]
    stddevs_pos = [np.std(intensity[dh_mask]) for intensity in intensities_pos]
    ptvs_pos = [np.max(intensity[dh_mask]) - np.min(intensity[dh_mask]) for intensity in intensities_pos]

    averages_neg = [np.mean(intensity[dh_mask]) for intensity in intensities_neg]
    stddevs_neg = [np.std(intensity[dh_mask]) for intensity in intensities_neg]
    ptvs_neg = [np.max(intensity[dh_mask]) - np.min(intensity[dh_mask]) for intensity in intensities_neg]

    efields = np.vstack([efields_pos, efields_neg])
    intensities = np.vstack([intensities_pos, intensities_neg])
    averages = np.vstack([averages_pos, averages_neg])
    stddevs = np.vstack([stddevs_pos, stddevs_neg])
    ptvs = np.vstack([ptvs_pos, ptvs_neg])

    return averages, stddevs, ptvs, efields, intensities


def analyze_probe_set_wvln_cubes(cfg, dmlist, dpv_list, dh_mask, indices=[0, 1, 2]):
    """
    Analyze a set of probes across multiple wavelength indices and return datacubes.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting
     dpv_list: list of 3 DM probe voltages
     dh_mask: boolean mask for the dark hole region, in the focal plane
     indices: list of wavelength indices to analyze (default: [0, 1, 2])

    Returns:
     averages_cube: 3D array (n_indices, 2, 3) - average DH intensities
     stddevs_cube: 3D array (n_indices, 2, 3) - standard deviations
     ptvs_cube: 3D array (n_indices, 2, 3) - peak-to-valley values
     efields_cube: 4D array (n_indices, 2, 3, *field_shape) - electric fields
     intensities_cube: 4D array (n_indices, 2, 3, *field_shape) - intensities
    """

    # Initialize lists to store results for each index
    all_averages = []
    all_stddevs = []
    all_ptvs = []
    all_efields = []
    all_intensities = []

    # Loop over the specified indices
    for ind in indices:
        print(f"Analyzing wavelength index {ind}...")

        # Call analyze_probe_set for this index
        averages, stddevs, ptvs, efields, intensities = analyze_probe_set(
            cfg, dmlist, dpv_list, dh_mask, ind
        )

        # Store results
        all_averages.append(averages)
        all_stddevs.append(stddevs)
        all_ptvs.append(ptvs)
        all_efields.append(efields)
        all_intensities.append(intensities)

    # Convert lists to numpy arrays (datacubes)
    averages_cube = np.array(all_averages)  # Shape: (n_indices, 2, 3)
    stddevs_cube = np.array(all_stddevs)    # Shape: (n_indices, 2, 3)
    ptvs_cube = np.array(all_ptvs)          # Shape: (n_indices, 2, 3)
    efields_cube = np.array(all_efields)    # Shape: (n_indices, 2, 3, field_height, field_width)
    intensities_cube = np.array(all_intensities)  # Shape: (n_indices, 2, 3, field_height, field_width)

    print(f"Analysis complete. Datacube shapes:")
    print(f"  Averages: {averages_cube.shape}")
    print(f"  Stddevs: {stddevs_cube.shape}")
    print(f"  PTVs: {ptvs_cube.shape}")
    print(f"  EFields: {efields_cube.shape}")
    print(f"  Intensities: {intensities_cube.shape}")

    return averages_cube, stddevs_cube, ptvs_cube, efields_cube, intensities_cube


def save_wvln_cubes_to_disk(averages_cube, stddevs_cube, ptvs_cube,
                          output_path, prefix="probe_analysis"):
    """
    Save the datacubes returned by analyze_probe_set_wvln_cubes to disk as JSON files.

    Arguments:
     averages_cube, stddevs_cube, ptvs_cube: 3D arrays from analyze_probe_set_wvln_cubes
     output_path: directory path where files will be saved
     prefix: filename prefix (default: "probe_analysis")
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Convert numpy arrays to Python lists for JSON serialization
    averages_data = averages_cube.tolist()
    stddevs_data = stddevs_cube.tolist()
    ptvs_data = ptvs_cube.tolist()

    # Create metadata for the JSON files
    metadata = {
        "shape": averages_cube.shape,
        "description": "3D datacube with dimensions (wavelength_index, probing_sequence, probe_number)",
        "wavelength_indices": [0, 1, 2],
        "probing_sequences": ["positive", "negative"],
        "probe_numbers": [0, 1, 2]
    }

    # Save each datacube as a JSON file
    averages_json = {
        "metadata": metadata,
        "data": averages_data
    }
    with open(os.path.join(output_path, f"{prefix}_averages.json"), 'w') as f:
        json.dump(averages_json, f, indent=2)
    print(f"Saved: {prefix}_averages.json")

    stddevs_json = {
        "metadata": metadata,
        "data": stddevs_data
    }
    with open(os.path.join(output_path, f"{prefix}_stddevs.json"), 'w') as f:
        json.dump(stddevs_json, f, indent=2)
    print(f"Saved: {prefix}_stddevs.json")

    ptvs_json = {
        "metadata": metadata,
        "data": ptvs_data
    }
    with open(os.path.join(output_path, f"{prefix}_ptvs.json"), 'w') as f:
        json.dump(ptvs_json, f, indent=2)
    print(f"Saved: {prefix}_ptvs.json")

    print(f"All data saved to: {output_path}")


def plot_probe_ni_vs_wvln(averages_cube):
    """
    Generate analysis plots for probe data from a single probe set across all wavelengths.

    Arguments:
     averages_cube: 3D array of average DH intensities (normalized)
                   Shape: (n_wavelengths, 2, 3)
                   - 0th index: wavelength indices
                   - 1st index: probe sequence (0=positive, 1=negative)
                   - 2nd index: probe number (0, 1, 2)
    """

    # Validate input
    if not isinstance(averages_cube, np.ndarray):
        raise TypeError("averages_cube must be a numpy array")
    if len(averages_cube.shape) != 3:
        raise ValueError(f"averages_cube must be 3D, got shape {averages_cube.shape}")
    if averages_cube.shape[1] != 2 or averages_cube.shape[2] != 3:
        raise ValueError(f"averages_cube must have shape (n_wavelengths, 2, 3), got {averages_cube.shape}")

    wavelengths_nm = np.array([546, 575, 604])
    n_wavelengths = averages_cube.shape[0]

    # Ensure we have the right number of wavelengths
    if len(wavelengths_nm) < n_wavelengths:
        # Extend wavelengths if needed
        wavelengths_nm = np.linspace(546, 604, n_wavelengths)

    # Define colors for each probe number (0, 1, 2)
    probe_colors = ['blue', 'green', 'red']
    probe_labels = ['Probe 0', 'Probe 1', 'Probe 2']

    # Define markers for each probe sequence (positive/negative)
    markers = ['+', 'x']  # plus for positive, x for negative
    probe_sequence_labels = ['Positive', 'Negative']

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each data point
    for wvl_idx in range(n_wavelengths):
        for probe_sequence in range(2):  # 0=positive, 1=negative
            for probe_num in range(3):  # 0, 1, 2
                ax.scatter(wavelengths_nm[wvl_idx], averages_cube[wvl_idx, probe_sequence, probe_num],
                           color=probe_colors[probe_num], marker=markers[probe_sequence],
                           s=150, alpha=0.8, linewidth=2, edgecolors='black')

    # Create custom legends
    # Legend for markers (probe sequence: positive/negative)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(markers, probe_sequence_labels)):
        marker_handles.append(plt.Line2D([0], [0], marker=marker, color='gray',
                                         linestyle='None', markersize=12, label=label,
                                         markeredgewidth=2))

    # Legend for colors (probe numbers: 0, 1, 2)
    color_handles = []
    for i, (color, label) in enumerate(zip(probe_colors, probe_labels)):
        color_handles.append(plt.Line2D([0], [0], marker='o', color=color,
                                        linestyle='None', markersize=10,
                                        label=label, markeredgecolor='black'))

    # Add legends with positioning to avoid data overlap
    legend1 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.7), title='Probe Sequence')
    legend2 = ax.legend(handles=color_handles, loc='center left', bbox_to_anchor=(1.02, 0.3), title='Probe Number')
    ax.add_artist(legend1)  # Add the first legend back

    # Set labels and title
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Average DH intensity of probe (normalized)', fontsize=12)
    ax.set_title('Average probed DH intensity vs wavelength', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set x-axis limits with some padding
    x_padding = (wavelengths_nm.max() - wavelengths_nm.min()) * 0.1
    ax.set_xlim(wavelengths_nm.min() - x_padding, wavelengths_nm.max() + x_padding)

    plt.tight_layout()
    # Adjust layout to make room for legends outside the plot area
    plt.subplots_adjust(right=0.75)
    plt.show()

    return fig, ax


def create_gaussian_probe_sets_sigma_sweep(modelpath, cfgfile, dmlist, sigma_range, sigma_step,
                                         deltax_act_list=[13, 13, 14], deltay_act_list=[8, 9, 9],
                                         ni_desired=1e-5, lod_min=2.8, lod_max=209.7, ind=1):
    """
    Create multiple Gaussian probe sets with different sigma parameters.

    Arguments:
     modelpath: path to the HOWFSC model directory
     cfgfile: configuration file
     dmlist: list of DMs for current DM setting
     sigma_range: tuple of (min_sigma, max_sigma) for the sigma parameter sweep
     sigma_step: step size for sigma parameter sweep
     deltax_act_list: list of x-offsets for the 3 probes in actuator widths (default: [13, 13, 14])
     deltay_act_list: list of y-offsets for the 3 probes in actuator widths (default: [8, 9, 9])
     ni_desired: desired mean normalized intensity of probes (default: 1e-5)
     lod_min: minimum lambda/D for scoring region (default: 2.8)
     lod_max: maximum lambda/D for scoring region (default: 209.7)
     ind: wavelength index to use for optimization (default: 1)

    Returns:
     dpv_sets_dict: dictionary where keys are sigma values and values are lists of 3 dpv arrays
     sigma_values: array of sigma values used
     metadata: dictionary containing parameters used for generation
    """

    # Generate sigma values
    sigma_min, sigma_max = sigma_range
    sigma_values = np.arange(sigma_min, sigma_max + sigma_step, sigma_step)

    homf_dict = loadyaml(cfgfile)
    diam_pupil_pix = homf_dict['sls'][1]['epup']['pdp']
    dmreg_dm1 = homf_dict['dms']['DM1']['registration']
    dact = diam_pupil_pix / dmreg_dm1['ppact_cx']

    # Get tie map for usable actuators
    tiemap_file = os.path.join(modelpath, homf_dict['dms']['DM1']['voltages']['tiefn'])
    tiemap = fits.getdata(tiemap_file)
    NACT = homf_dict['dms']['DM1']['registration']['nact']
    usable_act_map = np.zeros((NACT, NACT), dtype=bool)
    usable_act_map[tiemap == 0] = True

    dpv_sets_dict = {}
    metadata = {
        'sigma_range': sigma_range,
        'sigma_step': sigma_step,
        'deltax_act_list': deltax_act_list,
        'deltay_act_list': deltay_act_list,
        'ni_desired': ni_desired,
        'lod_min': lod_min,
        'lod_max': lod_max,
        'wavelength_index': ind,
        'dact': dact
    }

    print(f"Creating Gaussian probe sets for sigma range {sigma_min:.2f} to {sigma_max:.2f} with step {sigma_step:.2f}")
    print(f"This will generate {len(sigma_values)} probe sets, each with 3 probes")

    for sigma in sigma_values:
        print(f"\nGenerating probe set for sigma = {sigma:.2f}")
        dpv_list = []

        for probe_idx, (deltax_act, deltay_act) in enumerate(zip(deltax_act_list, deltay_act_list)):
            print(f"  Creating probe {probe_idx}: x={deltax_act}, y={deltay_act}")

            probe_tuple = make_dmrel_probe_gaussian(
                cfg=cfg, dmlist=dmlist, dact=dact,
                xcenter=deltax_act, ycenter=deltay_act, sigma=sigma,
                target=ni_desired, lod_min=lod_min, lod_max=lod_max,
                ind=ind, maxiter=5
            )

            dpv = probe_tuple[0]

            # Apply usable actuator mask if available
            if usable_act_map is not None:
                dpv = usable_act_map * dpv
            else:
                raise ValueError("Usable actuator map could not be loaded. Check the tie map file path and contents.")

            dpv_list.append(dpv)

        dpv_sets_dict[sigma] = dpv_list
        print(f"  Completed probe set for sigma = {sigma:.2f}")
        print(f"  Probe voltages (min, max) for this set: {np.min(dpv_list), np.max(dpv_list)}")

    # Access DH mask, assuming that all probes used the same mask
    dh_mask = probe_tuple[2]

    print(f"\nCompleted generation of {len(sigma_values)} probe sets")
    return dpv_sets_dict, sigma_values, dh_mask, metadata


def save_gaussian_probe_sets_sigma_sweep(dpv_sets_dict, sigma_values, dh_mask, metadata, output_path,
                                         mode='nfov_band1', dark_hole='360deg', prefix='gaussian_sigma_sweep'):
    """
    Save Gaussian probe sets from sigma sweep to disk as FITS files.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep
     sigma_values: array of sigma values
     dh_mask: boolean mask for the dark hole region, in the focal plane
     metadata: metadata dictionary
     output_path: directory path where files will be saved
     mode: coronagraph mode string for filename
     dark_hole: dark hole configuration for filename
     prefix: filename prefix
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    ni_desired = metadata['ni_desired']

    for sigma in sigma_values:
        dpv_list = dpv_sets_dict[sigma]

        for probe_idx, dpv in enumerate(dpv_list):
            # Create filename following the pattern from write_gaussian_probes.py
            deltax = metadata['deltax_act_list'][probe_idx]
            deltay = metadata['deltay_act_list'][probe_idx]

            filename = f"dmrel_{mode}_{dark_hole}_ni{ni_desired:.0e}_x{deltax}_y{deltay}_sigma{sigma:.2f}_gauss{probe_idx}.fits"
            filepath = os.path.join(output_path, filename)

            fits.writeto(filepath, dpv, overwrite=True)
            print(f"Saved: {filename}")

    # Save DH mask as a FITS file
    fits.writeto(os.path.join(output_path, "dh_mask.fits"), dh_mask.astype(np.float32), overwrite=True)

    # Save metadata as JSON
    metadata_enhanced = metadata.copy()
    metadata_enhanced['sigma_values'] = sigma_values.tolist()
    metadata_enhanced['mode'] = mode
    metadata_enhanced['dark_hole'] = dark_hole
    metadata_enhanced['n_probe_sets'] = len(sigma_values)
    metadata_enhanced['n_probes_per_set'] = len(metadata['deltax_act_list'])

    metadata_file = os.path.join(output_path, f"{prefix}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata_enhanced, f, indent=2)
    print(f"Saved metadata: {prefix}_metadata.json")

    print(f"All probe sets saved to: {output_path}")


def plot_sigma_sweep_mean_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                   wavelength_indices=[0, 1, 2], probe_indices=[0, 1, 2]):
    """
    Create a plot showing average DH intensity vs sigma for different wavelengths and probes.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep
     sigma_values: array of sigma values used
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])

    Returns:
     fig, ax: matplotlib figure and axes objects
    """

    # Get wavelength information
    wavelengths_nm = [546, 575, 604]  # Assuming these are the standard wavelengths
    colors = ['blue', 'green', 'red']  # Colors for different wavelengths

    # Initialize storage for results
    results = {}

    print(f"Analyzing sigma sweep data for {len(sigma_values)} sigma values...")

    # Analyze each sigma value
    for sigma_idx, sigma in enumerate(sigma_values):
        print(f"Processing sigma = {sigma:.2f} ({sigma_idx+1}/{len(sigma_values)})")

        dpv_list = dpv_sets_dict[sigma]

        # Analyze this probe set across all wavelengths
        try:
            averages_cube, _, _, _, _ = analyze_probe_set_wvln_cubes(
                cfg, dmlist, dpv_list, dh_mask, indices=wavelength_indices
            )

            results[sigma] = averages_cube

        except Exception as e:
            print(f"  Warning: Failed to analyze sigma {sigma:.2f}: {e}")
            continue

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Line styles for positive and negative probes
    line_styles = ['-', '--']  # solid for positive, dashed for negative
    probe_sequence_labels = ['positive', 'negative']

    # Markers for each probe index
    probe_markers = ['o', 's', '*']  # circle for probe 0, square for probe 1, star for probe 2
    probe_marker_labels = ['Probe 0', 'Probe 1', 'Probe 2']

    # Plot data for each wavelength, probe sequence, and probe index
    for wvl_idx, (wvl, color) in enumerate(zip(wavelength_indices, colors)):
        wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wvl}"

        for probe_sequence in range(2):  # 0=positive, 1=negative
            for probe_idx in probe_indices:
                # Extract data for this configuration
                sigma_plot = []
                intensity_plot = []

                for sigma in sigma_values:
                    if sigma in results:
                        try:
                            # Get intensity for this wavelength, probe sequence, and probe index
                            intensity = results[sigma][wvl_idx, probe_sequence, probe_idx]
                            sigma_plot.append(sigma)
                            intensity_plot.append(intensity)
                        except (IndexError, KeyError):
                            continue

                if len(sigma_plot) > 0:
                    # Create label
                    label = f"{wvl_nm}nm, probe {probe_idx}, {probe_sequence_labels[probe_sequence]}"

                    # Plot the line with probe-specific marker
                    marker = probe_markers[probe_idx] if probe_idx < len(probe_markers) else 'o'
                    ax.plot(sigma_plot, intensity_plot,
                           color=color, linestyle=line_styles[probe_sequence],
                           alpha=0.7, linewidth=1.5, label=label, marker=marker, markersize=4)

    # Customize the plot
    ax.set_xlabel('Gaussian σ (in actuator pitch)', fontsize=12)
    ax.set_ylabel('Average intensity in DH', fontsize=12)
    ax.set_title('Average normalized intensity in DH vs. Gaussian FWHM', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Format y-axis to show scientific notation if needed
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Legend for line styles (probe sequence)
    line_style_handles = []
    for i, (style, label) in enumerate(zip(line_styles, probe_sequence_labels)):
        line_style_handles.append(Line2D([0], [0], color='gray', linestyle=style,
                                        linewidth=1.5, label=label))

    # Legend for markers (probe indices)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(probe_markers, probe_marker_labels)):
        marker_handles.append(Line2D([0], [0], color='gray', marker=marker,
                                   linestyle='None', markersize=6, label=label))

    # Legend for colors (wavelengths)
    color_handles = []
    for i, color in enumerate(colors[:len(wavelength_indices)]):
        wvl_nm = wavelengths_nm[i] if i < len(wavelengths_nm) else f"λ{wavelength_indices[i]}"
        color_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"{wvl_nm}nm"))

    # Create three separate legends
    legend1 = ax.legend(handles=line_style_handles, loc='center left', bbox_to_anchor=(1.02, 0.85),
                       title='Probe sequence', fontsize=9)
    legend2 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.65),
                       title='Probe index', fontsize=9)
    legend3 = ax.legend(handles=color_handles, loc='center left', bbox_to_anchor=(1.02, 0.45),
                       title='Wavelength', fontsize=9)

    # Add the first two legends back (matplotlib only keeps the last one by default)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    plt.show()

    return fig, ax


def plot_sigma_sweep_stdev_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                    metadata, dpv_list_sincs=None, wavelength_indices=[0, 1, 2],
                                    probe_indices=[0, 1, 2], data_out=None):
    """
    Create a plot showing standard deviation of DH intensity as percentage of ni_desired vs sigma.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep or load function
     sigma_values: array of sigma values used
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     metadata: metadata dictionary containing ni_desired
     dpv_list_sincs: list of sinc probe voltages to analyze (optional)
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])
     data_out: path to store analysis results for sinc probes (optional, used if dpv_list_sincs is provided)

    Returns:
     fig, ax: matplotlib figure and axes objects
    """

    # Get wavelength information and plotting parameters (same as original function)
    wavelengths_nm = [546, 575, 604]
    colors = ['blue', 'green', 'red']  # Colors for different wavelengths

    # Get ni_desired from metadata for percentage calculation
    ni_desired = metadata['ni_desired']

    # Initialize storage for results
    results = {}

    print(f"Analyzing sigma sweep standard deviation data for {len(sigma_values)} sigma values...")

    # Analyze each sigma value
    for sigma_idx, sigma in enumerate(sigma_values):
        print(f"Processing sigma = {sigma:.2f} ({sigma_idx+1}/{len(sigma_values)})")

        dpv_list = dpv_sets_dict[sigma]

        # Analyze this probe set across all wavelengths
        _, stddevs_cube, _, _, _ = analyze_probe_set_wvln_cubes(
            cfg, dmlist, dpv_list, dh_mask, indices=wavelength_indices
        )

        results[sigma] = stddevs_cube

    # Analyze sinc probes if provided
    sinc_probe_means = None  # Will store means for each probe individually [wavelength_idx, probe_idx]
    if dpv_list_sincs is not None:
        print("\nAnalyzing sinc probes across all three wavelengths...")

        # Analyze sinc probes across all wavelengths
        _, stddevs_sinc_cube, _, _, _ = analyze_probe_set_wvln_cubes(
            cfg, dmlist, dpv_list_sincs, dh_mask, indices=wavelength_indices
        )

        # Get ni_desired for percentage calculation
        ni_desired = metadata['ni_desired']

        # Calculate mean stddev for each probe individually, averaged over probe sequences
        sinc_probe_means = []

        probe_sequence_labels = ['positive', 'negative']

        # Prepare data for saving
        analysis_text = []
        analysis_text.append("="*60)
        analysis_text.append("SINC PROBE STANDARD DEVIATION ANALYSIS RESULTS")
        analysis_text.append("="*60)
        analysis_text.append(f"Target normalized intensity (ni_desired): {ni_desired:.2e}")
        analysis_text.append("")

        # Prepare CSV data
        csv_data = []
        csv_headers = ['Wavelength_nm', 'Wavelength_Index', 'Probe_Sequence', 'Probe_Index', 'Stddev_Raw', 'Stddev_Percent']
        csv_data.append(csv_headers)

        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            analysis_text.append(f"Wavelength {wvl_nm} nm (index {wavelength_indices[wvl_idx]}):")
            analysis_text.append("-" * 40)

            # Calculate means for each probe individually at this wavelength
            probe_means_this_wvl = []

            for probe_idx in range(3):  # For each probe
                # Get stddevs for this probe across both sequences (pos and neg)
                probe_stddevs = []
                for probe_sequence in range(2):  # 0=positive, 1=negative
                    stdev_raw = stddevs_sinc_cube[wvl_idx, probe_sequence, probe_idx]
                    stdev_percent = (stdev_raw / ni_desired) * 100
                    probe_stddevs.append(stdev_percent)

                    # Add to CSV data
                    csv_data.append([wvl_nm, wavelength_indices[wvl_idx], probe_sequence_labels[probe_sequence],
                                   probe_idx, stdev_raw, stdev_percent])

                # Calculate mean for this probe (averaged over positive/negative)
                probe_mean = np.mean(probe_stddevs)
                probe_means_this_wvl.append(probe_mean)
                analysis_text.append(f"  Probe {probe_idx} (avg over pos/neg): {probe_mean:.2f}% of target")

            sinc_probe_means.append(probe_means_this_wvl)
            analysis_text.append("")

        analysis_text.append("="*60)
        analysis_text.append("SINC PROBE INDIVIDUAL MEANS (averaged over probe sequences):")
        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            analysis_text.append(f"  {wvl_nm}nm:")
            for probe_idx in range(3):
                analysis_text.append(f"    Probe {probe_idx}: {sinc_probe_means[wvl_idx][probe_idx]:.2f}% of target")
        analysis_text.append("="*60)

        # Save to txt file
        with open(os.path.join(data_out, 'sinc_probe_stddev_intensity.txt'), 'w') as f:
            for line in analysis_text:
                f.write(line + '\n')

        # Save to CSV file
        with open(os.path.join(data_out, 'sinc_probe_stddev_intensity.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Line styles for positive and negative probes (same as original function)
    line_styles = ['-', '--']  # solid for positive, dashed for negative
    probe_sequence_labels = ['positive', 'negative']

    # Markers for each probe index (same as original function)
    probe_markers = ['o', 's', '*']  # circle for probe 0, square for probe 1, star for probe 2
    probe_marker_labels = ['Probe 0', 'Probe 1', 'Probe 2']

    # Plot data for each wavelength, probe sequence, and probe index
    for wvl_idx, (wvl, color) in enumerate(zip(wavelength_indices, colors)):
        wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wvl}"

        for probe_sequence in range(2):  # 0=positive, 1=negative
            for probe_idx in probe_indices:
                # Extract data for this configuration
                sigma_plot = []
                stdev_plot = []

                for sigma in sigma_values:
                    if sigma in results:
                        # Get standard deviation for this wavelength, probe sequence, and probe index
                        stdev = results[sigma][wvl_idx, probe_sequence, probe_idx]
                        # Convert to percentage of ni_desired
                        stdev_percent = (stdev / ni_desired) * 100
                        sigma_plot.append(sigma)
                        stdev_plot.append(stdev_percent)

                if len(sigma_plot) > 0:
                    # Create label
                    label = f"{wvl_nm}nm, probe {probe_idx}, {probe_sequence_labels[probe_sequence]}"

                    # Plot the line with probe-specific marker
                    marker = probe_markers[probe_idx] if probe_idx < len(probe_markers) else 'o'
                    ax.plot(sigma_plot, stdev_plot,
                           color=color, linestyle=line_styles[probe_sequence],
                           alpha=0.7, linewidth=1.5, label=label, marker=marker, markersize=4)

    # Calculate and plot mean lines for each wavelength
    mean_lines_data = {}  # Store for sinc probe intersection calculation

    for wvl_idx, (wvl, color) in enumerate(zip(wavelength_indices, colors)):
        wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wvl}"

        # Collect all data points for this wavelength across all probe sequences and indices
        sigma_all = []
        stdev_all = []

        for probe_sequence in range(2):
            for probe_idx in probe_indices:
                for sigma in sigma_values:
                    if sigma in results:
                        stdev = results[sigma][wvl_idx, probe_sequence, probe_idx]
                        stdev_percent = (stdev / ni_desired) * 100
                        sigma_all.append(sigma)
                        stdev_all.append(stdev_percent)

        if len(sigma_all) > 0:
            # Calculate mean at each sigma value
            unique_sigmas = sorted(set(sigma_all))
            mean_stdevs = []

            for sigma in unique_sigmas:
                # Get all stdev values for this sigma
                sigma_stdevs = [stdev_all[i] for i, s in enumerate(sigma_all) if s == sigma]
                mean_stdevs.append(np.mean(sigma_stdevs))

            # Plot mean line with 'x' markers in grey
            ax.plot(unique_sigmas, mean_stdevs, color='black', linestyle='-', linewidth=3,
                   marker='x', markersize=8, alpha=0.6,
                   label=f"{wvl_nm}nm MEAN")

            # Store for sinc probe intersection calculation
            mean_lines_data[wvl_idx] = (unique_sigmas, mean_stdevs)

    # Plot sinc probe means at intersection points with Gaussian mean lines
    intersection_sigmas = []  # Store sigma values for legend display [(wvl_nm, probe_idx, sigma_val), ...]
    if sinc_probe_means is not None and len(sinc_probe_means) > 0:
        # Diamond marker styles: solid line for probe 0, dashed for probes 1 and 2
        diamond_line_styles = ['-', '--', '--']  # solid for probe 0, dashed for others

        for wvl_idx in range(len(wavelength_indices)):
            if wvl_idx in mean_lines_data:
                wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
                color = colors[wvl_idx]

                # Get Gaussian mean line data for intersection calculation
                unique_sigmas, mean_stdevs = mean_lines_data[wvl_idx]

                # Plot each probe individually
                for probe_idx in range(3):
                    sinc_mean = sinc_probe_means[wvl_idx][probe_idx]

                    # Find closest intersection point
                    differences = [abs(stdev - sinc_mean) for stdev in mean_stdevs]
                    closest_idx = np.argmin(differences)
                    intersection_sigma = unique_sigmas[closest_idx]
                    intersection_stdev = mean_stdevs[closest_idx]  # Y-value ON the Gaussian curve
                    intersection_sigmas.append((wvl_nm, probe_idx, intersection_sigma))

                    # Plot sinc probe mean at intersection with different diamond styles
                    # Use intersection_stdev (Gaussian curve Y-value) instead of sinc_mean
                    line_style = diamond_line_styles[probe_idx]
                    if line_style == '-':  # solid diamond
                        ax.plot(intersection_sigma, intersection_stdev, color=color, marker='D',
                               markersize=12, markerfacecolor='white', markeredgecolor=color,
                               markeredgewidth=3, linestyle='None',
                               label=f"{wvl_nm}nm P{probe_idx} SINC", zorder=10)
                    else:  # dashed diamond - simulate with markeredgewidth and alpha
                        ax.plot(intersection_sigma, intersection_stdev, color=color, marker='D',
                               markersize=12, markerfacecolor='white', markeredgecolor=color,
                               markeredgewidth=2, linestyle='None', alpha=0.7,
                               label=f"{wvl_nm}nm P{probe_idx} SINC", zorder=10)

    # Customize the plot
    ax.set_xlabel('Gaussian σ (in actuator pitch)', fontsize=12)
    ax.set_ylabel('Stddev of DH intensity (in % of target NI)', fontsize=12)
    ax.set_title('Stddev of DH intensity vs. Gaussian FWHM', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legends to explain the plot elements (same structure as original function)

    # Legend for line styles (probe sequence)
    line_style_handles = []
    for i, (style, label) in enumerate(zip(line_styles, probe_sequence_labels)):
        line_style_handles.append(Line2D([0], [0], color='gray', linestyle=style,
                                        linewidth=1.5, label=label))

    # Legend for markers (probe indices)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(probe_markers, probe_marker_labels)):
        marker_handles.append(Line2D([0], [0], color='gray', marker=marker,
                                   linestyle='None', markersize=6, label=label))

    # Legend for colors (wavelengths)
    color_handles = []
    for i, color in enumerate(colors[:len(wavelength_indices)]):
        wvl_nm = wavelengths_nm[i] if i < len(wavelengths_nm) else f"λ{wavelength_indices[i]}"
        color_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"{wvl_nm}nm"))

    # Legend for additional plot elements (mean lines and sinc probes)
    additional_handles = [
        Line2D([0], [0], color='gray', linewidth=3, marker='x', markersize=8,
               label='Gaussian Mean'),
        Line2D([0], [0], color='gray', marker='D', markersize=12,
               markerfacecolor='white', markeredgecolor='gray', markeredgewidth=3,
               linestyle='None', label='Sinc Mean')
    ]

    # Create four separate legends
    legend1 = ax.legend(handles=line_style_handles, loc='center left', bbox_to_anchor=(1.02, 0.9),
                       title='Probe Sequence', fontsize=9)
    legend2 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.75),
                       title='Probe Index', fontsize=9)
    legend3 = ax.legend(handles=color_handles, loc='center left', bbox_to_anchor=(1.02, 0.55),
                       title='Wavelength', fontsize=9)
    legend4 = ax.legend(handles=additional_handles, loc='center left', bbox_to_anchor=(1.02, 0.35),
                       title='Statistics', fontsize=9)

    # Add the first three legends back (matplotlib only keeps the last one by default)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)

    # Add sigma intersection values underneath the legends
    if sinc_probe_means is not None and intersection_sigmas:
        sigma_text = "Sinc σ intersections:\n"
        # Group by wavelength and probe for organized display
        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            sigma_text += f"{wvl_nm}: "
            probe_sigmas = []
            for wvl_name, probe_idx, sigma_val in intersection_sigmas:
                if wvl_name == wvl_nm:
                    probe_sigmas.append(f"P{probe_idx}={sigma_val:.2f}")
            sigma_text += ", ".join(probe_sigmas) + "\n"

        # Add text below the legends
        ax.text(1.02, 0.15, sigma_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='lightgray', alpha=0.8))

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    plt.show()

    return fig, ax


def plot_sigma_sweep_ptv_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                  metadata, dpv_list_sincs=None, wavelength_indices=[0, 1, 2],
                                  probe_indices=[0, 1, 2], data_out=None):
    """
    Create a plot showing peak-to-valley (max-min) of DH intensity as percentage of ni_desired vs sigma.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep or load function
     sigma_values: array of sigma values used
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     metadata: metadata dictionary containing ni_desired
     dpv_list_sincs: list of sinc probe voltages to analyze (optional)
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])
     data_out: path to store analysis results for sinc probes (optional, used if dpv_list_sincs is provided)

    Returns:
     fig, ax: matplotlib figure and axes objects
    """

    # Get wavelength information and plotting parameters (same as original function)
    wavelengths_nm = [546, 575, 604]
    colors = ['blue', 'green', 'red']  # Colors for different wavelengths

    # Get ni_desired from metadata for percentage calculation
    ni_desired = metadata['ni_desired']

    # Initialize storage for results
    results = {}

    print(f"Analyzing sigma sweep peak-to-valley data for {len(sigma_values)} sigma values...")

    # Analyze each sigma value
    for sigma_idx, sigma in enumerate(sigma_values):
        print(f"Processing sigma = {sigma:.2f} ({sigma_idx+1}/{len(sigma_values)})")

        dpv_list = dpv_sets_dict[sigma]

        # Analyze this probe set across all wavelengths
        _, _, ptvs_cube, _, _ = analyze_probe_set_wvln_cubes(
            cfg, dmlist, dpv_list, dh_mask, indices=wavelength_indices
        )

        results[sigma] = ptvs_cube

    # Analyze sinc probes if provided
    sinc_probe_means = None  # Will store means for each probe individually [wavelength_idx, probe_idx]
    if dpv_list_sincs is not None:
        # Analyze sinc probes across all wavelengths
        _, _, ptvs_sinc_cube, _, _ = analyze_probe_set_wvln_cubes(
            cfg, dmlist, dpv_list_sincs, dh_mask, indices=wavelength_indices
        )

        # Get ni_desired for percentage calculation
        ni_desired = metadata['ni_desired']

        # Calculate mean ptv for each probe individually, averaged over probe sequences
        sinc_probe_means = []

        probe_sequence_labels = ['positive', 'negative']

        # Prepare data for saving
        analysis_text = []
        analysis_text.append("="*60)
        analysis_text.append("SINC PROBE PEAK-TO-VALLEY ANALYSIS RESULTS")
        analysis_text.append("="*60)
        analysis_text.append(f"Target normalized intensity (ni_desired): {ni_desired:.2e}")
        analysis_text.append("")

        # Prepare CSV data
        csv_data = []
        csv_headers = ['Wavelength_nm', 'Wavelength_Index', 'Probe_Sequence', 'Probe_Index', 'PTV_Raw', 'PTV_Percent']
        csv_data.append(csv_headers)

        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            analysis_text.append(f"Wavelength {wvl_nm} nm (index {wavelength_indices[wvl_idx]}):")
            analysis_text.append("-" * 40)

            # Calculate means for each probe individually at this wavelength
            probe_means_this_wvl = []

            for probe_idx in range(3):  # For each probe
                # Get ptvs for this probe across both sequences (pos and neg)
                probe_ptvs = []
                for probe_sequence in range(2):  # 0=positive, 1=negative
                    ptv_raw = ptvs_sinc_cube[wvl_idx, probe_sequence, probe_idx]
                    ptv_percent = (ptv_raw / ni_desired) * 100
                    probe_ptvs.append(ptv_percent)

                    # Add to CSV data
                    csv_data.append([wvl_nm, wavelength_indices[wvl_idx], probe_sequence_labels[probe_sequence],
                                   probe_idx, ptv_raw, ptv_percent])

                # Calculate mean for this probe (averaged over positive/negative)
                probe_mean = np.mean(probe_ptvs)
                probe_means_this_wvl.append(probe_mean)
                analysis_text.append(f"  Probe {probe_idx} (avg over pos/neg): {probe_mean:.2f}% of target")

            sinc_probe_means.append(probe_means_this_wvl)
            analysis_text.append("")

        analysis_text.append("="*60)
        analysis_text.append("SINC PROBE INDIVIDUAL MEANS (averaged over probe sequences):")
        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            analysis_text.append(f"  {wvl_nm}nm:")
            for probe_idx in range(3):
                analysis_text.append(f"    Probe {probe_idx}: {sinc_probe_means[wvl_idx][probe_idx]:.2f}% of target")
        analysis_text.append("="*60)

        # Save to txt file
        with open(os.path.join(data_out, 'sinc_probe_ptv_intensity.txt'), 'w') as f:
            for line in analysis_text:
                f.write(line + '\n')

        # Save to CSV file
        import csv
        with open(os.path.join(data_out, 'sinc_probe_ptv_intensity.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)


    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Line styles for positive and negative probes (same as original function)
    line_styles = ['-', '--']  # solid for positive, dashed for negative
    probe_sequence_labels = ['positive', 'negative']

    # Markers for each probe index (same as original function)
    probe_markers = ['o', 's', '*']  # circle for probe 0, square for probe 1, star for probe 2
    probe_marker_labels = ['Probe 0', 'Probe 1', 'Probe 2']

    # Plot data for each wavelength, probe sequence, and probe index
    for wvl_idx, (wvl, color) in enumerate(zip(wavelength_indices, colors)):
        wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wvl}"

        for probe_sequence in range(2):  # 0=positive, 1=negative
            for probe_idx in probe_indices:
                # Extract data for this configuration
                sigma_plot = []
                ptv_plot = []

                for sigma in sigma_values:
                    if sigma in results:
                        # Get peak-to-valley for this wavelength, probe sequence, and probe index
                        ptv = results[sigma][wvl_idx, probe_sequence, probe_idx]
                        # Convert to percentage of ni_desired
                        ptv_percent = (ptv / ni_desired) * 100
                        sigma_plot.append(sigma)
                        ptv_plot.append(ptv_percent)

                if len(sigma_plot) > 0:
                    # Create label
                    label = f"{wvl_nm}nm, probe {probe_idx}, {probe_sequence_labels[probe_sequence]}"

                    # Plot the line with probe-specific marker
                    marker = probe_markers[probe_idx] if probe_idx < len(probe_markers) else 'o'
                    ax.plot(sigma_plot, ptv_plot,
                           color=color, linestyle=line_styles[probe_sequence],
                           alpha=0.7, linewidth=1.5, label=label, marker=marker, markersize=4)

    # Calculate and plot mean lines for each wavelength
    mean_lines_data = {}  # Store for sinc probe intersection calculation

    for wvl_idx, (wvl, color) in enumerate(zip(wavelength_indices, colors)):
        wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wvl}"

        # Collect all data points for this wavelength across all probe sequences and indices
        sigma_all = []
        ptv_all = []

        for probe_sequence in range(2):
            for probe_idx in probe_indices:
                for sigma in sigma_values:
                    if sigma in results:
                        ptv = results[sigma][wvl_idx, probe_sequence, probe_idx]
                        ptv_percent = (ptv / ni_desired) * 100
                        sigma_all.append(sigma)
                        ptv_all.append(ptv_percent)

        if len(sigma_all) > 0:
            # Calculate mean at each sigma value
            unique_sigmas = sorted(set(sigma_all))
            mean_ptvs = []

            for sigma in unique_sigmas:
                # Get all ptv values for this sigma
                sigma_ptvs = [ptv_all[i] for i, s in enumerate(sigma_all) if s == sigma]
                mean_ptvs.append(np.mean(sigma_ptvs))

            # Plot mean line with 'x' markers in black
            ax.plot(unique_sigmas, mean_ptvs, color='black', linestyle='-', linewidth=3,
                   marker='x', markersize=8, alpha=0.6,
                   label=f"{wvl_nm}nm MEAN")

            # Store for sinc probe intersection calculation
            mean_lines_data[wvl_idx] = (unique_sigmas, mean_ptvs)

    # Plot sinc probe means at intersection points with Gaussian mean lines
    intersection_sigmas = []  # Store sigma values for legend display [(wvl_nm, probe_idx, sigma_val), ...]
    if sinc_probe_means is not None and len(sinc_probe_means) > 0:
        # Diamond marker styles: solid line for probe 0, dashed for probes 1 and 2
        diamond_line_styles = ['-', '--', '--']  # solid for probe 0, dashed for others

        for wvl_idx in range(len(wavelength_indices)):
            if wvl_idx in mean_lines_data:
                wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
                color = colors[wvl_idx]

                # Get Gaussian mean line data for intersection calculation
                unique_sigmas, mean_ptvs = mean_lines_data[wvl_idx]

                # Plot each probe individually
                for probe_idx in range(3):
                    sinc_mean = sinc_probe_means[wvl_idx][probe_idx]

                    # Find closest intersection point
                    differences = [abs(ptv - sinc_mean) for ptv in mean_ptvs]
                    closest_idx = np.argmin(differences)
                    intersection_sigma = unique_sigmas[closest_idx]
                    intersection_ptv = mean_ptvs[closest_idx]  # Y-value ON the Gaussian curve
                    intersection_sigmas.append((wvl_nm, probe_idx, intersection_sigma))

                    # Plot sinc probe mean at intersection with different diamond styles
                    # Use intersection_ptv (Gaussian curve Y-value) instead of sinc_mean
                    line_style = diamond_line_styles[probe_idx]
                    if line_style == '-':  # solid diamond
                        ax.plot(intersection_sigma, intersection_ptv, color=color, marker='D',
                               markersize=12, markerfacecolor='white', markeredgecolor=color,
                               markeredgewidth=3, linestyle='None',
                               label=f"{wvl_nm}nm P{probe_idx} SINC", zorder=10)
                    else:  # dashed diamond - simulate with markeredgewidth and alpha
                        ax.plot(intersection_sigma, intersection_ptv, color=color, marker='D',
                               markersize=12, markerfacecolor='white', markeredgecolor=color,
                               markeredgewidth=2, linestyle='None', alpha=0.7,
                               label=f"{wvl_nm}nm P{probe_idx} SINC", zorder=10)

    # Customize the plot
    ax.set_xlabel('Gaussian σ (in actuator pitch)', fontsize=12)
    ax.set_ylabel('Peak-to-valley of DH intensity (in % of target NI)', fontsize=12)
    ax.set_title('Peak-to-valley of DH intensity vs. Gaussian FWHM', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend for line styles (probe sequence)
    line_style_handles = []
    for i, (style, label) in enumerate(zip(line_styles, probe_sequence_labels)):
        line_style_handles.append(Line2D([0], [0], color='gray', linestyle=style,
                                        linewidth=1.5, label=label))

    # Legend for markers (probe indices)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(probe_markers, probe_marker_labels)):
        marker_handles.append(Line2D([0], [0], color='gray', marker=marker,
                                   linestyle='None', markersize=6, label=label))

    # Legend for colors (wavelengths)
    color_handles = []
    for i, color in enumerate(colors[:len(wavelength_indices)]):
        wvl_nm = wavelengths_nm[i] if i < len(wavelengths_nm) else f"λ{wavelength_indices[i]}"
        color_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"{wvl_nm}nm"))

    # Legend for additional plot elements (mean lines and sinc probes)
    additional_handles = [
        Line2D([0], [0], color='gray', linewidth=3, marker='x', markersize=8,
               label='Gaussian Mean'),
        Line2D([0], [0], color='gray', marker='D', markersize=12,
               markerfacecolor='white', markeredgecolor='gray', markeredgewidth=3,
               linestyle='None', label='Sinc Mean')
    ]

    # Create four separate legends
    legend1 = ax.legend(handles=line_style_handles, loc='center left', bbox_to_anchor=(1.02, 0.9),
                       title='Probe Sequence', fontsize=9)
    legend2 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.75),
                       title='Probe Index', fontsize=9)
    legend3 = ax.legend(handles=color_handles, loc='center left', bbox_to_anchor=(1.02, 0.55),
                       title='Wavelength', fontsize=9)
    legend4 = ax.legend(handles=additional_handles, loc='center left', bbox_to_anchor=(1.02, 0.35),
                       title='Statistics', fontsize=9)

    # Add the first three legends back (matplotlib only keeps the last one by default)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)

    # Add sigma intersection values underneath the legends
    if sinc_probe_means is not None and intersection_sigmas:
        sigma_text = "Sinc σ intersections:\n"
        # Group by wavelength and probe for organized display
        for wvl_idx in range(len(wavelength_indices)):
            wvl_nm = wavelengths_nm[wvl_idx] if wvl_idx < len(wavelengths_nm) else f"λ{wavelength_indices[wvl_idx]}"
            sigma_text += f"{wvl_nm}: "
            probe_sigmas = []
            for wvl_name, probe_idx, sigma_val in intersection_sigmas:
                if wvl_name == wvl_nm:
                    probe_sigmas.append(f"P{probe_idx}={sigma_val:.2f}")
            sigma_text += ", ".join(probe_sigmas) + "\n"

        # Add text below the legends
        ax.text(1.02, 0.15, sigma_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='lightgray', alpha=0.8))

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    plt.show()

    return fig, ax


def load_gaussian_probe_sets_sigma_sweep(input_path, prefix='gaussian_sigma_sweep',
                                        mode='nfov_band1', dark_hole='360deg'):
    """
    Load Gaussian probe sets from sigma sweep saved on disk.

    Arguments:
     input_path: directory path where files are saved
     prefix: filename prefix used when saving (default: 'gaussian_sigma_sweep')
     mode: coronagraph mode string used in filename
     dark_hole: dark hole configuration used in filename

    Returns:
     dpv_sets_dict: dictionary where keys are sigma values and values are lists of 3 dpv arrays
     sigma_values: array of sigma values
     metadata: dictionary containing parameters used for generation
    """

    # Load metadata first
    metadata_file = os.path.join(input_path, f"{prefix}_metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    sigma_values = np.array(metadata['sigma_values'])
    ni_desired = metadata['ni_desired']
    deltax_act_list = metadata['deltax_act_list']
    deltay_act_list = metadata['deltay_act_list']

    print(f"Loading Gaussian probe sets from: {input_path}")
    print(f"Found metadata for {len(sigma_values)} sigma values")

    # Load the DH mask
    dh_mask = fits.getdata(os.path.join(input_path, "dh_mask.fits")).astype(bool)

    # Load the DPV arrays for each sigma value
    dpv_sets_dict = {}

    for sigma in sigma_values:
        print(f"Loading probe set for sigma = {sigma:.2f}")
        dpv_list = []

        for probe_idx, (deltax, deltay) in enumerate(zip(deltax_act_list, deltay_act_list)):
            # Reconstruct filename following the same pattern used in saving
            filename = f"dmrel_{mode}_{dark_hole}_ni{ni_desired:.0e}_x{deltax}_y{deltay}_sigma{sigma:.2f}_gauss{probe_idx}.fits"
            filepath = os.path.join(input_path, filename)

            try:
                dpv = fits.getdata(filepath)
                dpv_list.append(dpv)
                print(f"  Loaded: {filename}")
            except Exception as e:
                raise FileNotFoundError(f"  Error loading {filename}: {e}")

        dpv_sets_dict[sigma] = dpv_list
        print(f"  Completed loading probe set for sigma = {sigma:.2f}")
        print(f"  Probe voltages (min, max) for this set: {np.min(dpv_list), np.max(dpv_list)}")

    print(f"Completed loading {len(sigma_values)} probe sets from disk")
    return dpv_sets_dict, sigma_values, dh_mask, metadata


def plot_dm_amplitude_vs_sigma(dpv_sets_dict, sigma_values, dpv_list_sincs, cfg, dmlist, dh_mask,
                               metadata, wavelength_indices=[0, 1, 2], probe_indices=[0, 1, 2]):
    """
    Create a plot showing probe amplitude on DM in peak-to-valley (volts) vs Gaussian FWHM (sigma)
    in units of actuator pitch, with sinc-sinc-sine probe data overlaid.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep
     sigma_values: array of sigma values used for Gaussian probes
     dpv_list_sincs: list of sinc-sinc-sine probe voltage arrays
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     metadata: metadata dictionary containing ni_desired
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])

    Returns:
     fig, ax: matplotlib figure and axes objects
    """

    # Get wavelength information and plotting parameters
    wavelengths_nm = [546, 575, 604]
    colors = ['blue', 'green', 'red']  # Colors for different wavelengths

    # Markers for different probes
    probe_markers = ['o', 's', '*']  # circle, square, star

    # Initialize storage for Gaussian probe results
    gaussian_results = {}

    print(f"Analyzing DM amplitude data for {len(sigma_values)} Gaussian sigma values...")

    # Analyze each Gaussian sigma value
    for sigma_idx, sigma in enumerate(sigma_values):
        print(f"Processing Gaussian sigma = {sigma:.2f} ({sigma_idx+1}/{len(sigma_values)})")

        dpv_list = dpv_sets_dict[sigma]

        # Calculate peak-to-valley amplitude for each probe in this set
        probe_ptvs = []
        for dpv in dpv_list:
            ptv_volts = np.max(dpv) - np.min(dpv)
            probe_ptvs.append(ptv_volts)

        # Analyze probe intensities for each wavelength
        wvln_results = []
        for wvl_idx in wavelength_indices:
            averages, stddevs, ptvs, efields, intensities = analyze_probe_set(
                cfg, dmlist, dpv_list, dh_mask, wvl_idx
            )
            wvln_results.append(averages)

        # Store results: [wavelength_idx, pos/neg_sequence, probe_idx] -> (amplitude_ptv, intensity_avg)
        wvln_results = np.array(wvln_results)  # shape: (n_wvl, 2, n_probes)
        gaussian_results[sigma] = {
            'amplitudes': probe_ptvs,
            'intensities': wvln_results
        }

    # Analyze sinc-sinc-sine probes
    print("\nAnalyzing sinc-sinc-sine probes...")
    sinc_amplitudes = []
    sinc_intensities_by_wvl = []

    for dpv in dpv_list_sincs:
        ptv_volts = np.max(dpv) - np.min(dpv)
        sinc_amplitudes.append(ptv_volts)

    # Get sinc probe intensities for each wavelength
    for wvl_idx in wavelength_indices:
        averages, stddevs, ptvs, efields, intensities = analyze_probe_set(
            cfg, dmlist, dpv_list_sincs, dh_mask, wvl_idx
        )
        sinc_intensities_by_wvl.append(averages)

    sinc_intensities_by_wvl = np.array(sinc_intensities_by_wvl)  # shape: (n_wvl, 2, n_probes)

    # Calculate equivalent sigma for sinc-sinc-sine probes based on intensity matching
    print("\nCalculating equivalent sigma for sinc-sinc-sine probes...")
    sinc_equivalent_sigmas = []

    for probe_idx in range(len(dpv_list_sincs)):
        # For each sinc probe, find the Gaussian sigma that gives closest intensity match
        # Use the average intensity from positive and negative sequences at middle wavelength
        wvl_middle_idx = len(wavelength_indices) // 2
        sinc_avg_intensity = np.mean(sinc_intensities_by_wvl[wvl_middle_idx, :, probe_idx])

        best_sigma = None
        min_diff = float('inf')

        for sigma in sigma_values:
            gauss_avg_intensity = np.mean(gaussian_results[sigma]['intensities'][wvl_middle_idx, :, probe_idx])
            diff = abs(gauss_avg_intensity - sinc_avg_intensity)
            if diff < min_diff:
                min_diff = diff
                best_sigma = sigma

        sinc_equivalent_sigmas.append(best_sigma)
        print(f"  Sinc probe {probe_idx}: equivalent sigma = {best_sigma:.2f}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot Gaussian probe data
    for wvl_idx, wvl_nm in enumerate(wavelengths_nm):
        if wvl_idx not in wavelength_indices:
            continue

        color = colors[wvl_idx]

        for probe_idx in probe_indices:
            if probe_idx >= len(probe_markers):
                continue

            # Extract data for this configuration
            sigma_plot = []
            amplitude_plot = []

            for sigma in sigma_values:
                if sigma in gaussian_results:
                    try:
                        amplitude = gaussian_results[sigma]['amplitudes'][probe_idx]
                        sigma_plot.append(sigma)
                        amplitude_plot.append(amplitude)
                    except (IndexError, KeyError):
                        continue

            if len(sigma_plot) > 0:
                # Create label for Gaussian data
                label = f"Gaussian {wvl_nm}nm, probe {probe_idx}"

                # Plot the Gaussian curve
                marker = probe_markers[probe_idx]
                ax.plot(sigma_plot, amplitude_plot,
                       color=color, linestyle='-', alpha=0.7, linewidth=2,
                       marker=marker, markersize=6, label=label)

    # Overlay sinc-sinc-sine probe data
    for probe_idx in probe_indices:
        if probe_idx >= len(sinc_amplitudes):
            continue

        equivalent_sigma = sinc_equivalent_sigmas[probe_idx]
        amplitude = sinc_amplitudes[probe_idx]

        # Plot sinc probe as a single point with distinct styling
        marker = probe_markers[probe_idx]
        # Make star marker three times bigger since it appears smaller than circle/square
        marker_size = 300 if marker == '*' else 100
        ax.scatter(equivalent_sigma, amplitude,
                  color='black', s=marker_size, marker=marker,
                  edgecolors='white', linewidth=2,
                  label=f"Sinc probe {probe_idx} (σ≈{equivalent_sigma:.2f})",
                  zorder=10)

    # Customize the plot
    ax.set_xlabel('Gaussian FWHM (σ) [actuator pitch]', fontsize=12)
    ax.set_ylabel('DM Probe Amplitude Peak-to-Valley [V]', fontsize=12)
    ax.set_title('DM Probe Amplitude vs Gaussian Sigma with Sinc-Sinc-Sine Overlay', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Set reasonable axis limits
    if len(sigma_values) > 0:
        ax.set_xlim(min(sigma_values) - 0.1, max(sigma_values) + 0.3)

    plt.tight_layout()
    plt.subplots_adjust(right=0.7)
    plt.show()

    return fig, ax


if __name__ == '__main__':
    mode = 'nfov_band1'
    dark_hole = '360deg'
    analysis_path = '/Users/ilaginja/Nextcloud/Areas/RomanCPP/alternate_probes/probe_comparison/active_analysis'

    # Load probes and DH mask from the analysis path
    dh_mask = fits.getdata(os.path.join(analysis_path, 'dh_mask.fits')).astype(bool)

    dpv_list_gaussians = []   # Gaussian probes
    ### 1e-5
    # dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y8_gauss1.fits')))
    # dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y9_gauss2.fits')))
    # dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x14_y9_gauss3.fits')))

    ### 1e-7 with sigma=1 from sigma sweep
    dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'sigma_sweep_1e-7', 'dmrel_nfov_band1_360deg_ni1e-07_x13_y8_sigma1.50_gauss0.fits')))
    dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'sigma_sweep_1e-7', 'dmrel_nfov_band1_360deg_ni1e-07_x13_y8_sigma1.50_gauss0.fits')))
    dpv_list_gaussians.append(fits.getdata(os.path.join(analysis_path, 'sigma_sweep_1e-7', 'dmrel_nfov_band1_360deg_ni1e-07_x13_y8_sigma1.50_gauss0.fits')))

    dpv_list_sincs = []   # Baseline sinc-sinc-sine probes, originally scaled to 1e-5
    scale = 0.13
    cos = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_cos.fits') * scale
    dpv_list_sincs.append(cos)
    sinlr = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinlr.fits') * scale
    dpv_list_sincs.append(sinlr)
    sinud = fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinud.fits') * scale
    dpv_list_sincs.append(sinud)

    print(f"  Probe voltages (min, max) for cos: {np.min(cos), np.max(cos)} -> PtV = {np.max(cos) - np.min(cos)}")
    print(f"  Probe voltages (min, max) for sinlr: {np.min(sinlr), np.max(sinlr)} -> PtV = {np.max(sinlr) - np.min(sinlr)}")
    print(f"  Probe voltages (min, max) for sinud: {np.min(sinud), np.max(sinud)} -> PtV = {np.max(sinud) - np.min(sinud)}")

    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps

    ################## BASIC ANALYSIS OF TWO SETS OF PROBES ##################
    # # Analyze probe sets for datacubes across wavelength indices [0, 1, 2]
    # print("Analyzing Gaussian probes...")
    # averages_cube_gaussian, stddevs_cube1, ptvs_cube1, efields_cube1, intensities_cube1 = analyze_probe_set_wvln_cubes(
    #     cfg, dmlist, dpv_list_gaussians, dh_mask, indices=[0, 1, 2]
    # )

    # print("\nAnalyzing Sinusoidal probes...")
    # averages_cube_sines, stddevs_cube2, ptvs_cube2, efields_cube2, intensities_cube2 = analyze_probe_set_wvln_cubes(
    #     cfg, dmlist, dpv_list_sincs, dh_mask, indices=[0, 1, 2]
    # )

    # # Save wvln cubes to disk
    # print(f"\nSaving Gaussian probe datacubes...")
    # save_wvln_cubes_to_disk(averages_cube_gaussian, stddevs_cube1, ptvs_cube1, analysis_path, prefix="gaussian_probes")

    # print(f"\nSaving Sinusoidal probe datacubes...")
    # save_wvln_cubes_to_disk(averages_cube_sines, stddevs_cube2, ptvs_cube2, analysis_path, prefix="sinusoidal_probes")

    # # Plot comparison of averages across wavelengths
    # print("\nGenerating comparison plots...")
    # plot_probe_ni_vs_wvln(averages_cube_gaussian)

    ################## SIGMA SWEEP ##################
    output_path = os.path.join(analysis_path, 'sigma_sweep_1e-7')  # Path where data is saved

    # # Option 1: Create new Gaussian probe sets with sigma sweep (uncomment to generate new data)
    # sigma_range = (0.5, 2.0)  # Range for sigma values
    # sigma_step = 0.1         # Step size for sigma sweep

    # print("\nCreating Gaussian probe sets with sigma sweep...")
    # dpv_sets_dict, sigma_values, dh_mask, metadata = create_gaussian_probe_sets_sigma_sweep(
    #     modelpath, cfgfile, dmlist, sigma_range, sigma_step,
    #     deltax_act_list=[13, 13, 14], deltay_act_list=[8, 9, 9],
    #     ni_desired=1e-7, lod_min=2.8, lod_max=209.7, ind=1
    # )
    #
    # print("\nSaving Gaussian probe sets to disk...")
    # save_gaussian_probe_sets_sigma_sweep(dpv_sets_dict, sigma_values, dh_mask, metadata, output_path,
    #                                      mode='nfov_band1', dark_hole='360deg', prefix='gaussian_sigma_sweep')

    # Option 2: Load existing Gaussian probe sets from disk
    print("\nLoading Gaussian probe sets from disk...")
    dpv_sets_dict, sigma_values, dh_mask, metadata = load_gaussian_probe_sets_sigma_sweep(
        output_path, prefix='gaussian_sigma_sweep', mode='nfov_band1', dark_hole='360deg'
    )

    # # Plot sigma sweep mean DH intensity analysis
    # print("\nGenerating sigma sweep analysis plot...")
    # plot_sigma_sweep_mean_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
    #                                wavelength_indices=[0, 1, 2])

    # # Plot sigma sweep standard deviation analysis
    # print("\nGenerating sigma sweep standard deviation analysis plot...")
    # plot_sigma_sweep_stdev_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
    #                                 metadata, dpv_list_sincs, wavelength_indices=[0, 1, 2],
    #                                 data_out=analysis_path)

    # Plot sigma sweep peak-to-valley analysis
    print("\nGenerating sigma sweep peak-to-valley analysis plot...")
    plot_sigma_sweep_ptv_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                  metadata, dpv_list_sincs, wavelength_indices=[0, 1, 2],
                                  data_out=analysis_path)

    # # Plot DM amplitude vs sigma with sinc-sinc-sine overlay
    # print("\nGenerating DM amplitude vs sigma analysis plot with sinc-sinc-sine overlay...")
    # plot_dm_amplitude_vs_sigma(dpv_sets_dict, sigma_values, dpv_list_sincs, cfg, dmlist, dh_mask,
    #                            metadata, wavelength_indices=[0, 1, 2])

