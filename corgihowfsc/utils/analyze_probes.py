from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check
from howfsc.util.prop_tools import efield, open_efield

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files

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

    # Calculate the electric field for each probe, positive and negative
    efields_pos = []
    intensities_pos = []
    efields_neg = []
    intensities_neg = []
    for i, probe_v in enumerate(dpv_list):
        probed_field_pos = efield(cfg, [dmlist[0] + probe_v, dmlist[1]], ind)
        efields_pos.append(probed_field_pos / np.sqrt(ipeak))
        intensities_pos.append(np.abs(probed_field_pos)**2 / ipeak)

        probed_field_neg = efield(cfg, [dmlist[0] - probe_v, dmlist[1]], ind)
        efields_neg.append(probed_field_neg / np.sqrt(ipeak))
        intensities_neg.append(np.abs(probed_field_neg)**2 / ipeak)

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
                   - 1st index: probe type (0=positive, 1=negative)
                   - 2nd index: probe number (0, 1, 2)
    """

    # Validate input
    if not isinstance(averages_cube, np.ndarray):
        raise TypeError("averages_cube must be a numpy array")
    if len(averages_cube.shape) != 3:
        raise ValueError(f"averages_cube must be 3D, got shape {averages_cube.shape}")
    if averages_cube.shape[1] != 2 or averages_cube.shape[2] != 3:
        raise ValueError(f"averages_cube must have shape (n_wavelengths, 2, 3), got {averages_cube.shape}")

    wavelengths_um = np.array([546, 575, 604])
    n_wavelengths = averages_cube.shape[0]

    # Ensure we have the right number of wavelengths
    if len(wavelengths_um) < n_wavelengths:
        # Extend wavelengths if needed
        wavelengths_um = np.linspace(546, 604, n_wavelengths)

    # Define colors for each probe number (0, 1, 2)
    probe_colors = ['blue', 'green', 'red']
    probe_labels = ['Probe 0', 'Probe 1', 'Probe 2']

    # Define markers for each probe type (positive/negative)
    markers = ['+', 'x']  # plus for positive, x for negative
    probe_type_labels = ['Positive', 'Negative']

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each data point
    for wvl_idx in range(n_wavelengths):
        for probe_type in range(2):  # 0=positive, 1=negative
            for probe_num in range(3):  # 0, 1, 2
                ax.scatter(wavelengths_um[wvl_idx], averages_cube[wvl_idx, probe_type, probe_num],
                           color=probe_colors[probe_num], marker=markers[probe_type],
                           s=150, alpha=0.8, linewidth=2, edgecolors='black')

    # Create custom legends
    # Legend for markers (probe types: positive/negative)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(markers, probe_type_labels)):
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
    legend1 = ax.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(1.02, 0.7), title='Probe Type')
    legend2 = ax.legend(handles=color_handles, loc='center left', bbox_to_anchor=(1.02, 0.3), title='Probe Number')
    ax.add_artist(legend1)  # Add the first legend back

    # Set labels and title
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Average DH intensity of probe (normalized)', fontsize=12)
    ax.set_title('Average probed DH intensity vs wavelength', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set x-axis limits with some padding
    x_padding = (wavelengths_um.max() - wavelengths_um.min()) * 0.1
    ax.set_xlim(wavelengths_um.min() - x_padding, wavelengths_um.max() + x_padding)

    plt.tight_layout()
    # Adjust layout to make room for legends outside the plot area
    plt.subplots_adjust(right=0.75)
    plt.show()

    return fig, ax


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
    dpv_list2.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_sin90_rot0.fits')))
    dpv_list2.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_sin150_rot0.fits')))
    dpv_list2.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_sin210_rot90.fits')))

    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps

    # Analyze probe sets for datacubes across wavelength indices [0, 1, 2]
    print("Analyzing Gaussian probes...")
    averages_cube1, stddevs_cube1, ptvs_cube1, efields_cube1, intensities_cube1 = analyze_probe_set_wvln_cubes(
        cfg, dmlist, dpv_list1, dh_mask, indices=[0, 1, 2]
    )

    print("\nAnalyzing Sinusoidal probes...")
    averages_cube2, stddevs_cube2, ptvs_cube2, efields_cube2, intensities_cube2 = analyze_probe_set_wvln_cubes(
        cfg, dmlist, dpv_list2, dh_mask, indices=[0, 1, 2]
    )

    # Save wvln cubes to disk
    print(f"\nSaving Gaussian probe datacubes...")
    save_wvln_cubes_to_disk(averages_cube1, stddevs_cube1, ptvs_cube1, analysis_path, prefix="gaussian_probes")

    print(f"\nSaving Sinusoidal probe datacubes...")
    save_wvln_cubes_to_disk(averages_cube2, stddevs_cube2, ptvs_cube2, analysis_path, prefix="sinusoidal_probes")

    # Plot comparison of averages across wavelengths
    print("\nGenerating comparison plots...")
    plot_probe_ni_vs_wvln(averages_cube1)  # Plot Gaussian probes across all wavelengths
