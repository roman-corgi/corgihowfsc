from howfsc.model.mode import CoronagraphMode
import howfsc.util.check as check
from howfsc.util.prop_tools import efield, open_efield

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

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


def plot_probe_ni_vs_wvln(averages):
    """
    Generate analysis plots for probe data.

    Arguments:
     averages: 2x3 array of average DH intensities (normalized)
    """

    wavelengths_um = np.array([546, 575, 604])

    # Define colors for each column (wavelength)
    colors = ['blue', 'green', 'orange']

    # Define markers for each row (positive/negative probes)
    markers = ['o', 's']  # circle for row 0, square for row 1
    labels = ['Positive probe', 'Negative probe']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each data point
    for row in range(averages.shape[0]):
        for col in range(averages.shape[1]):
            ax.scatter(wavelengths_um[col], averages[row, col],
                      color=colors[col], marker=markers[row], s=100,
                      alpha=0.7, edgecolors='black', linewidth=1)

    # Create custom legend
    # Legend for markers (rows)
    marker_handles = []
    for i, (marker, label) in enumerate(zip(markers, labels)):
        marker_handles.append(plt.Line2D([0], [0], marker=marker, color='gray',
                                       linestyle='None', markersize=8, label=label))

    # Legend for colors (columns/wavelengths)
    color_handles = []
    for i, color in enumerate(colors):
        color_handles.append(plt.Line2D([0], [0], marker='o', color=color,
                                      linestyle='None', markersize=8,
                                      label=f'{wavelengths_um[i]:.3f} μm'))

    # Add legends
    legend1 = ax.legend(handles=marker_handles, loc='upper right', title='Probe Type')
    legend2 = ax.legend(handles=color_handles, loc='upper left', title='Wavelength')
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
    plt.show()


if __name__ == '__main__':
    mode = 'nfov_band1'
    dark_hole = '360deg'
    analysis_path = '/Users/ilaginja/Nextcloud/Areas/RomanCPP/alternate_probes/probe_comparison/active_analysis'
    ind = 1

    # Load probes and DH mask from the analysis path
    dh_mask = fits.getdata(os.path.join(analysis_path, 'dh_mask.fits')).astype(bool)

    dpv_list = []
    dpv_list.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y8_gauss1.fits')))
    dpv_list.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x13_y9_gauss2.fits')))
    dpv_list.append(fits.getdata(os.path.join(analysis_path, 'dmrel_nfov_band1_360deg_ni1e-05_x14_y9_gauss3.fits')))

    howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps

    averages, stddevs, ptvs, efields, intensities = analyze_probe_set(cfg, dmlist, dpv_list, dh_mask, ind)
    plot_probe_ni_vs_wvln(averages)

    print(averages)
    print(averages.shape)