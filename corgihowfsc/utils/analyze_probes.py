from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check
from howfsc.util.prop_tools import efield, open_efield
from howfsc.util.loadyaml import loadyaml

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files
from corgihowfsc.utils.cgi_prop_tools import make_dmrel_probe_gaussian

from astropy.io import fits
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

def plot_sigma_sweep_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
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

    # Add legends to explain the plot elements
    from matplotlib.lines import Line2D

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
                                   metadata, wavelength_indices=[0, 1, 2], probe_indices=[0, 1, 2]):
    """
    Create a plot showing standard deviation of DH intensity as percentage of ni_desired vs sigma.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep or load function
     sigma_values: array of sigma values used
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     metadata: metadata dictionary containing ni_desired
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])

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
        try:
            _, stddevs_cube, _, _, _ = analyze_probe_set_wvln_cubes(
                cfg, dmlist, dpv_list, dh_mask, indices=wavelength_indices
            )

            results[sigma] = stddevs_cube

        except Exception as e:
            print(f"  Warning: Failed to analyze sigma {sigma:.2f}: {e}")
            continue

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
                        try:
                            # Get standard deviation for this wavelength, probe sequence, and probe index
                            stdev = results[sigma][wvl_idx, probe_sequence, probe_idx]
                            # Convert to percentage of ni_desired
                            stdev_percent = (stdev / ni_desired) * 100
                            sigma_plot.append(sigma)
                            stdev_plot.append(stdev_percent)
                        except (IndexError, KeyError):
                            continue

                if len(sigma_plot) > 0:
                    # Create label
                    label = f"{wvl_nm}nm, probe {probe_idx}, {probe_sequence_labels[probe_sequence]}"

                    # Plot the line with probe-specific marker
                    marker = probe_markers[probe_idx] if probe_idx < len(probe_markers) else 'o'
                    ax.plot(sigma_plot, stdev_plot,
                           color=color, linestyle=line_styles[probe_sequence],
                           alpha=0.7, linewidth=1.5, label=label, marker=marker, markersize=4)

    # Customize the plot
    ax.set_xlabel('Gaussian σ (in actuator pitch)', fontsize=12)
    ax.set_ylabel('Stddev of DH intensity (in % of target NI)', fontsize=12)
    ax.set_title('Stddev of DH intensity vs. Gaussian FWHM', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legends to explain the plot elements (same structure as original function)
    from matplotlib.lines import Line2D

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
                       title='Probe Type', fontsize=9)
    legend2 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.65),
                       title='Probe Index', fontsize=9)
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


def plot_sigma_sweep_ptv_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                metadata, wavelength_indices=[0, 1, 2], probe_indices=[0, 1, 2]):
    """
    Create a plot showing peak-to-valley (max-min) of DH intensity as percentage of ni_desired vs sigma.

    Arguments:
     dpv_sets_dict: dictionary from create_gaussian_probe_sets_sigma_sweep or load function
     sigma_values: array of sigma values used
     cfg: CoronagraphMode object
     dmlist: list of DMs for current DM setting
     dh_mask: boolean mask for the dark hole region
     metadata: metadata dictionary containing ni_desired
     wavelength_indices: list of wavelength indices to analyze (default: [0, 1, 2])
     probe_indices: list of probe indices to plot (default: [0, 1, 2])

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
        try:
            _, _, ptvs_cube, _, _ = analyze_probe_set_wvln_cubes(
                cfg, dmlist, dpv_list, dh_mask, indices=wavelength_indices
            )

            results[sigma] = ptvs_cube

        except Exception as e:
            print(f"  Warning: Failed to analyze sigma {sigma:.2f}: {e}")
            continue

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
                        try:
                            # Get peak-to-valley for this wavelength, probe sequence, and probe index
                            ptv = results[sigma][wvl_idx, probe_sequence, probe_idx]
                            # Convert to percentage of ni_desired
                            ptv_percent = (ptv / ni_desired) * 100
                            sigma_plot.append(sigma)
                            ptv_plot.append(ptv_percent)
                        except (IndexError, KeyError):
                            continue

                if len(sigma_plot) > 0:
                    # Create label
                    label = f"{wvl_nm}nm, probe {probe_idx}, {probe_sequence_labels[probe_sequence]}"

                    # Plot the line with probe-specific marker
                    marker = probe_markers[probe_idx] if probe_idx < len(probe_markers) else 'o'
                    ax.plot(sigma_plot, ptv_plot,
                           color=color, linestyle=line_styles[probe_sequence],
                           alpha=0.7, linewidth=1.5, label=label, marker=marker, markersize=4)

    # Customize the plot
    ax.set_xlabel('Gaussian σ (in actuator pitch)', fontsize=12)
    ax.set_ylabel('Peak-to-valley of DH intensity (in % of target NI)', fontsize=12)
    ax.set_title('Peak-to-valley of DH intensity vs. Gaussian FWHM', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legends to explain the plot elements (same structure as original function)
    from matplotlib.lines import Line2D

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

            if not os.path.exists(filepath):
                print(f"  Warning: File not found: {filename}")
                # Create a zero array as fallback
                if len(dpv_list) > 0:
                    dpv = np.zeros_like(dpv_list[0])
                else:
                    # Use a reasonable default size
                    dpv = np.zeros((48, 48), dtype=np.float32)
                dpv_list.append(dpv)
                continue

            try:
                dpv = fits.getdata(filepath)
                dpv_list.append(dpv)
                print(f"  Loaded: {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                # Create a zero array as fallback
                if len(dpv_list) > 0:
                    dpv = np.zeros_like(dpv_list[0])
                else:
                    dpv = np.zeros((48, 48), dtype=np.float32)
                dpv_list.append(dpv)

        dpv_sets_dict[sigma] = dpv_list
        print(f"  Completed loading probe set for sigma = {sigma:.2f}")

    print(f"Completed loading {len(sigma_values)} probe sets from disk")
    return dpv_sets_dict, sigma_values, dh_mask, metadata


if __name__ == '__main__':
    mode = 'nfov_band1'
    dark_hole = '360deg'
    analysis_path = '/Users/ilaginja/Nextcloud/Areas/RomanCPP/alternate_probes/probe_comparison/active_analysis'

    # Load probes and DH mask from the analysis path
    dh_mask = fits.getdata(os.path.join(analysis_path, 'dh_mask.fits')).astype(bool)

    dpv_list1 = []
    dpv_list1.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y8_gauss1.fits')))
    dpv_list1.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y9_gauss2.fits')))
    dpv_list1.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x14_y9_gauss3.fits')))

    dpv_list2 = []
    dpv_list2.append(fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_cos.fits'))
    dpv_list2.append(fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinlr.fits'))
    dpv_list2.append(fits.getdata('/Users/ilaginja/repos/corgihowfsc/corgihowfsc/model/probes/nfov_dm_dmrel_4_1.0e-05_sinud.fits'))

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
    # averages_cube1, stddevs_cube1, ptvs_cube1, efields_cube1, intensities_cube1 = analyze_probe_set_wvln_cubes(
    #     cfg, dmlist, dpv_list1, dh_mask, indices=[0, 1, 2]
    # )

    # print("\nAnalyzing Sinusoidal probes...")
    # averages_cube2, stddevs_cube2, ptvs_cube2, efields_cube2, intensities_cube2 = analyze_probe_set_wvln_cubes(
    #     cfg, dmlist, dpv_list2, dh_mask, indices=[0, 1, 2]
    # )

    # # Save wvln cubes to disk
    # print(f"\nSaving Gaussian probe datacubes...")
    # save_wvln_cubes_to_disk(averages_cube1, stddevs_cube1, ptvs_cube1, analysis_path, prefix="gaussian_probes")
    #
    # print(f"\nSaving Sinusoidal probe datacubes...")
    # save_wvln_cubes_to_disk(averages_cube2, stddevs_cube2, ptvs_cube2, analysis_path, prefix="sinusoidal_probes")

    # # Plot comparison of averages across wavelengths
    # print("\nGenerating comparison plots...")
    # plot_probe_ni_vs_wvln(averages_cube2)

    ################## SIGMA SWEEP ##################
    output_path = os.path.join(analysis_path, 'sigma_sweep_1e-7')  # Path where data is saved

    # Option 1: Create new Gaussian probe sets with sigma sweep (uncomment to generate new data)
    sigma_range = (0.5, 2.0)  # Range for sigma values
    sigma_step = 0.1         # Step size for sigma sweep

    print("\nCreating Gaussian probe sets with sigma sweep...")
    dpv_sets_dict, sigma_values, dh_mask, metadata = create_gaussian_probe_sets_sigma_sweep(
        modelpath, cfgfile, dmlist, sigma_range, sigma_step,
        deltax_act_list=[13, 13, 14], deltay_act_list=[8, 9, 9],
        ni_desired=1e-7, lod_min=2.8, lod_max=209.7, ind=1
    )

    print("\nSaving Gaussian probe sets to disk...")
    save_gaussian_probe_sets_sigma_sweep(dpv_sets_dict, sigma_values, dh_mask, metadata, output_path,
                                         mode='nfov_band1', dark_hole='360deg', prefix='gaussian_sigma_sweep')

    # Option 2: Load existing Gaussian probe sets from disk
    # print("\nLoading Gaussian probe sets from disk...")
    # dpv_sets_dict, sigma_values, dh_mask, metadata = load_gaussian_probe_sets_sigma_sweep(
    #     output_path, prefix='gaussian_sigma_sweep', mode='nfov_band1', dark_hole='360deg'
    # )

    # Plot sigma sweep analysis
    print("\nGenerating sigma sweep analysis plot...")
    plot_sigma_sweep_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                             wavelength_indices=[0, 1, 2])

    # Plot sigma sweep standard deviation analysis
    print("\nGenerating sigma sweep standard deviation analysis plot...")
    plot_sigma_sweep_stdev_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                    metadata, wavelength_indices=[0, 1, 2])

    # Plot sigma sweep peak-to-valley analysis
    print("\nGenerating sigma sweep peak-to-valley analysis plot...")
    plot_sigma_sweep_ptv_analysis(dpv_sets_dict, sigma_values, cfg, dmlist, dh_mask,
                                  metadata, wavelength_indices=[0, 1, 2])

