import os
import csv
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np

from howfsc.util.gitl_tools import param_order_to_list
from howfsc.util.insertinto import insertinto
from howfsc.util.svd_spectrum import calc_svd_spectrum

import matplotlib
matplotlib.use('Agg')

import logging
log = logging.getLogger(__name__)

markers = ['o','p','d','+','^']
tab20c = cm.get_cmap('tab20c')
# colours = [tab20c(i) for i in range(20)]
colours = [tab20c(i * 4) for i in range(5)]  # 5 groups


def compute_xticks(n, max_ticks=15, threshold=25, values=None):
    """Return an array of x-tick positions for n iterations to avoid overcrowding.

    - If n <= threshold: return every tick (1..n).
    - Otherwise pick a step so that roughly <= max_ticks ticks are shown,
      rounding the step up to a 'nice' value from values.
    """
    if values is None:
        values = np.array([1, 2, 5, 10, 20, 50, 100, 200])

    if n <= threshold:
        step = 1
    else:
        raw_step = int(np.ceil(n / float(max_ticks)))
        idx = np.searchsorted(values, raw_step)
        step = int(values[idx]) if idx < len(values) else raw_step

    ticks = np.arange(1, n + 1, step)
    if ticks.size == 0 or ticks[-1] != n:
        ticks = np.append(ticks, n)
    return ticks

def save_outputs_iter(i, fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list, output_every_iter, pred_c, ni_lists, perfect_efield_list, jac, iteration_durations=None, debugging_dict=None, normalizer = None, true_exptime_list=None):
    """
    Save all outputs for a single HOWFSC iteration and update cumulative summary plots.

    Writes per-iteration data into a subdirectory ``iteration_{i+1:04d}/`` under
    ``os.path.dirname(fileout)``. Also updates cumulative output files (contrast
    plots, NI plot, CSV logs) in the top-level output directory on every call,
    unless ``output_every_iter`` is False and this is the final iteration.

    Parameters
    ----------
    i : int
        Zero-based index of the iteration to save. Used to index into
        ``framelistlist``, ``otherlist``, ``camlist``, ``dm1_list``, and
        ``dm2_list``, and to name the output subdirectory.
    fileout : str
        Absolute path to the top-level output FITS file. The directory portion
        is used as the root for all output files and subdirectories.
    cfg : CoronagraphMode
        Optical model object. Used to determine the number of wavelength
        channels (``cfg.sl_list``) and dark hole masks.
    camlist : list
        List of camera parameter triples accumulated across iterations. Each
        element is ``[gain_list, exptime_list, nframes_list]`` for one
        iteration. ``camlist[i][1]`` (exposure times) is saved to
        ``images.fits``.
    framelistlist : list
        List of framelists accumulated across iterations. Each element is the
        ``nlam * ndm`` list of 2D intensity images for one iteration.
        ``framelistlist[i]`` is saved to ``images.fits``.
    otherlist : list
        List of ``other`` dictionaries accumulated across iterations. Each
        element is a dict keyed by wavelength index ``j``, containing products
        such as ``meas_efield``, ``meas_intensity``, ``modul_intensity``, and
        ``unmodul_intensity`` for that iteration.
    measured_c : list of float
        Measured contrast values accumulated across iterations, one per
        iteration. Plotted and saved to ``measured_contrast.csv``.
    dm1_list : list of ndarray or None
        Absolute DM1 command arrays accumulated across iterations, one 48x48
        array per iteration. ``dm1_list[i]`` is saved to
        ``dm1_command.fits``. Pass None to skip saving DM1 state.
    dm2_list : list of ndarray or None
        Absolute DM2 command arrays accumulated across iterations, one 48x48
        array per iteration. ``dm2_list[i]`` is saved to
        ``dm2_command.fits``. Pass None to skip saving DM2 state.
    output_every_iter : bool
        If True, cumulative summary plots and CSVs are updated on every call.
        If False, they are only written on the last iteration
        (``i < len(framelistlist) - 1`` check).
    pred_c : list of float
        Predicted contrast values accumulated across iterations, one per
        iteration. Plotted and saved to ``predicted_contrast.csv``.
    ni_lists : dict
        Dictionary of normalized intensity (NI) metric lists accumulated across iterations. Keys are
        metric names (e.g. ``'ni_score'``, ``'ni_inner'``, ``'ni_outer'``);
        values are lists of floats, one per iteration. All keys are plotted
        together on the NI summary plot.
    perfect_efield_list : list or None
        Perfect (model) electric fields for this iteration, one 2D complex
        array per wavelength. Passed directly to ``refactor_efields`` and saved
        to ``perfect_efields.fits``. Pass None if perfect e-fields are not
        available.
    jac : ndarray
        3D real-valued DM Jacobian array of shape ``(2, Ndm, Npix)``, as
        produced by ``calcjacs()``. Real parts are in ``jac[0]``, imaginary
        parts in ``jac[1]``. Used for SVD spectrum computation.
    debugging_dict : dict or None, optional
        Dictionary of scalar debugging quantities for this iteration, as
        returned by ``howfsc_computation``. If provided, per-wavelength scalars
        are appended to ``debugging_history.csv`` via
        ``save_debugging_iteration``. Defaults to None.
    normalizer : object or None, optional
        The instantiated normalization process used
        to accurately convert raw detector frames into contrast unit.
        Propagated to `save_normalized_images_cube`. Defaults to None.
    true_exptime_list : list or 1D array, optional
        List of exact exposure times corresponding to each frame in `flist` for the
        current iteration. Used for the normalization of probed images in contrast unit, bypassing
        the offset `camlist`. Defaults to None.
    Returns
    -------
    efields_complex_array : ndarray
        Complex e-field estimates for this iteration, shape
        ``(nlam, nrow, ncol)``.
    perfect_efields_complex_array : ndarray
        Perfect (model) complex e-fields for this iteration, shape
        ``(n_perf_lam, nrow, ncol)``, where ``n_perf_lam`` may differ from
        ``nlam`` if a speedup mode (single central wavelength) is in use.

    Files Written
    -------------
    Top-level output directory (cumulative, updated each call):
      - ``contrast_vs_iteration.pdf`` : measured and predicted contrast vs. iteration
      - ``ni_vs_iteration.pdf``       : NI metrics vs. iteration
      - ``measured_contrast.csv``     : measured contrast values
      - ``predicted_contrast.csv``    : predicted contrast values
      - ``debugging_history.csv``     : per-wavelength debugging scalars (if debugging_dict provided)

    Per-iteration subdirectory ``iteration_{i+1:04d}/``:
      - ``images.fits``               : raw GITL frames and exposure times
      - ``intensity_total.fits``      : total (coherent + incoherent) intensity per wavelength
      - ``intensity_coherent.fits``   : coherent (modulated) intensity per wavelength
      - ``intensity_incoherent.fits`` : incoherent (unmodulated) intensity per wavelength
      - ``efield_estimations.fits``   : real and imaginary parts of estimated e-fields
      - ``perfect_efields.fits``      : real and imaginary parts of model (perfect) e-fields
      - ``svd_snorm.fits``            : singular values squared, normalized by the maximum, ordered largest to smallest
      - ``svd_iri.fits``              : power per singular-value mode, in the same order as svd_snorm
      - ``dm1_command.fits``          : absolute DM1 voltage command (if dm1_list provided)
      - ``dm2_command.fits``          : absolute DM2 voltage command (if dm2_list provided)
    """

    outpath = os.path.dirname(fileout)

    if output_every_iter or (not output_every_iter and i < len(framelistlist)-1):
        # Plot measured_c vs iteration
        plt.figure(layout="constrained")
        x_meas = np.arange(len(measured_c)) + 1
        x_pred = np.arange(len(pred_c)) + 2

        plt.plot(x_meas, measured_c, color='cornflowerblue', marker='o', label='measured')
        plt.plot(x_pred, pred_c, color='orchid', marker='+', label='predicted (PWP + compact[dE_efc])')
        plt.xlabel('Iteration')
        plt.ylabel('Measured Contrast')
        plt.semilogy()

        ticks = compute_xticks(len(measured_c))
        plt.xticks(ticks)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outpath, "contrast_vs_iteration.pdf"), bbox_inches='tight')
        plt.close()

        # Plot measured_c vs cumulative iteration duration
        if iteration_durations is not None and len(iteration_durations) > 0:
            fig, ax_bottom = plt.subplots(layout="constrained")
            cumulative_time = np.cumsum(iteration_durations) / 60  # convert seconds to minutes
            cumulative_time_with_overhead = np.cumsum(iteration_durations) / 60 + np.arange(1,
                                                                                            len(iteration_durations) + 1) * 60

            ax_bottom.plot(cumulative_time, measured_c, color='cornflowerblue', marker='o', label='measured')
            ax_bottom.set_xlabel('Spacecraft Time [minutes]')
            ax_bottom.set_ylabel('Measured Contrast')
            ax_bottom.semilogy()
            ax_bottom.legend(loc='best')

            ax_top = ax_bottom.twiny()
            ax_top.set_xlim(cumulative_time_with_overhead[0], cumulative_time_with_overhead[-1])
            ax_top.set_xlabel('GITL Time (w/comms overhead) [minutes]')

            plt.savefig(os.path.join(outpath, "contrast_vs_time.pdf"), bbox_inches='tight')
            plt.close()

        # NI plot
        plt.figure(layout="constrained")
        x_meas = np.arange(len(measured_c)) + 1
        plt.plot(x_meas, measured_c, color='cornflowerblue', marker='o', label='measured contrast')
        for index, key in enumerate(ni_lists.keys()):
            colour = colours[index % len(colours)]
            marker = markers[index % len(markers)]
            plt.plot(np.arange(len(ni_lists[key])) + 1, ni_lists[key], color=colour, marker=marker, label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Measured NI')
        plt.semilogy()

        ticks = compute_xticks(len(measured_c))
        plt.xticks(ticks)
        plt.legend(loc='best')
        plt.savefig(os.path.join(outpath, "ni_vs_iteration.pdf"), bbox_inches='tight')
        plt.close()

        # Save measured_c to a csv file
        np.savetxt(os.path.join(outpath, "measured_contrast.csv"), np.array(measured_c), delimiter=",",
                   header="Measured Contrast", comments="")
        np.savetxt(os.path.join(outpath, "predicted_contrast.csv"), np.array(pred_c), delimiter=",",
                   header="Predicted Contrast", comments="")

        if debugging_dict is not None:
            save_debugging_iteration(debugging_dict, i + 1, outpath, csv_path='debugging_history.csv')

    # Create iteration subdirectory
    # i = len(framelistlist)-1
    iterpath = os.path.join(outpath, f"iteration_{i + 1:04d}")
    if not os.path.exists(iterpath):
        os.makedirs(iterpath)

    # Initialize lists to collect e-field data across all iterations
    # all_efields_complex = []  # Will collect complex e-fields per iteration
    # all_perfect_efields_complex = []  # Will collect perfect complex e-fields per iteration

    # Saving separate intensity files per iteration
    flist = framelistlist[i]

    oitem = otherlist[i]

    # Re-define iterpath for this loop
    iterpath = os.path.join(outpath, f"iteration_{i + 1:04d}")

    # List for all intensities (current iteration)
    stack_total = []
    stack_coh = []
    stack_incoh = []

    for n in range(len(cfg.sl_list)):
        # Total intensity
        total_int = oitem[n].get('meas_intensity', np.zeros((1, 1)))

        # Coherent intensity
        coh_int = oitem[n].get('modul_intensity', np.zeros_like(total_int))

        # Incoherent intensity
        incoh_int = oitem[n].get('unmodul_intensity', np.zeros_like(total_int))

        # Stacking list
        stack_total.append(total_int)
        stack_coh.append(coh_int)
        stack_incoh.append(incoh_int)

    actual_exptime = true_exptime_list if true_exptime_list is not None else param_order_to_list(camlist[i][1])
    hdr = pyfits.Header()
    hdr['NLAM'] = len(cfg.sl_list)
    hdr['ITER'] = i + 1

    # Saving data of images
    prim = pyfits.PrimaryHDU(header=hdr)
    img_raw = pyfits.ImageHDU(flist, name='RAW_IMAGES')
    prev = pyfits.ImageHDU(actual_exptime, name='CAM_PARAMS')

    hdul_main = pyfits.HDUList([prim, img_raw, prev])
    hdul_main.writeto(os.path.join(iterpath, "images.fits"), overwrite=True)

    # Saving data of images in contrast units

    save_normalized_images_cube(
        iterpath=iterpath,
        flist=flist,
        true_exptime_list=actual_exptime,
        debugging_dict=debugging_dict,
        nlam=len(cfg.sl_list),
        hdr=hdr,
        cam_params_hdu=prev,
        normalizer=normalizer
    )

    # Saving total intensity
    prim_tot = pyfits.PrimaryHDU(header=hdr)
    img_tot = pyfits.ImageHDU(np.array(stack_total), name='TOTAL_INTENSITY')
    hdul_tot = pyfits.HDUList([prim_tot, img_tot])
    hdul_tot.writeto(os.path.join(iterpath, "intensity_total.fits"), overwrite=True)

    # Saving coherent intensity
    prim_coh = pyfits.PrimaryHDU(header=hdr)
    img_coh = pyfits.ImageHDU(np.array(stack_coh), name='COHERENT_INTENSITY')
    hdul_coh = pyfits.HDUList([prim_coh, img_coh])
    hdul_coh.writeto(os.path.join(iterpath, "intensity_coherent.fits"), overwrite=True)

    # Saving incoherent intensity
    prim_incoh = pyfits.PrimaryHDU(header=hdr)
    img_incoh = pyfits.ImageHDU(np.array(stack_incoh), name='INCOHERENT_INTENSITY')
    hdul_incoh = pyfits.HDUList([prim_incoh, img_incoh])
    hdul_incoh.writeto(os.path.join(iterpath, "intensity_incoherent.fits"), overwrite=True)

    # --- E-FIELD ESTIMATIONS ---
    efields_realimag, efields_complex_array, perfect_efields_realimag, perfect_efields_complex_array = refactor_efields(cfg, oitem, perfect_efield_list=perfect_efield_list)

    hdr_ef = pyfits.Header()
    hdr_ef['NLAM'] = len(cfg.sl_list)
    prim_ef = pyfits.PrimaryHDU(header=hdr_ef)
    img_ef = pyfits.ImageHDU(np.array(efields_realimag))
    hdul_ef = pyfits.HDUList([prim_ef, img_ef])
    hdul_ef.writeto(os.path.join(iterpath, "efield_estimations.fits"), overwrite=True)

    # --- PERFECT E-FIELDS ---

    hdr_pef = pyfits.Header()
    hdr_pef['NLAM'] = len(cfg.sl_list)
    prim_pef = pyfits.PrimaryHDU(header=hdr_pef)
    img_pef = pyfits.ImageHDU(np.array(perfect_efields_realimag))
    hdul_pef = pyfits.HDUList([prim_pef, img_pef])
    hdul_pef.writeto(os.path.join(iterpath, "perfect_efields.fits"), overwrite=True)

    # --- E-FIELD SVD ---
    e0list = [oitem[j]['meas_efield'] for j in range(len(cfg.sl_list))]
    snorm, iri = calc_svd_spectrum(jac, cfg, e0list)

    pyfits.writeto(os.path.join(iterpath, "svd_snorm.fits"), snorm.astype(np.float32), overwrite=True)
    pyfits.writeto(os.path.join(iterpath, "svd_iri.fits"), iri.astype(np.float32), overwrite=True)

    # --- DM STATES (PER ITERATION) ---
    if dm1_list is not None and i < len(dm1_list):
        pyfits.writeto(os.path.join(iterpath, "dm1_command.fits"), dm1_list[i], overwrite=True)

    if dm2_list is not None and i < len(dm2_list):
        pyfits.writeto(os.path.join(iterpath, "dm2_command.fits"), dm2_list[i], overwrite=True)

        log.info(f"Saved outputs (individual) for iteration {i + 1}")

    return efields_complex_array, perfect_efields_complex_array

def refactor_efields(cfg, oitem, perfect_efield_list=None):
    # --- E-FIELD ESTIMATIONS ---
    efields_realimag = []
    efields_complex = []
    for j in range(len(cfg.sl_list)):
        efields_realimag.append(np.real(oitem[j]['meas_efield']))
        efields_realimag.append(np.imag(oitem[j]['meas_efield']))
        efields_complex.append(oitem[j]['meas_efield'])

    # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
    efields_complex_array = np.stack(efields_complex, axis=0)

    # --- PERFECT E-FIELDS ---
    perfect_efields_realimag = []
    perfect_efields_complex = []

    num_perf_lams = len(perfect_efield_list) if perfect_efield_list is not None else len(cfg.sl_list)

    for j in range(num_perf_lams):
        if perfect_efield_list is not None:
            perf_efield = perfect_efield_list[j]
        else:
            perf_efield = oitem[j]['model_efield']
        perfect_efields_realimag.append(np.real(perf_efield))
        perfect_efields_realimag.append(np.imag(perf_efield))
        perfect_efields_complex.append(perf_efield)

    # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
    perfect_efields_complex_array = np.stack(perfect_efields_complex, axis=0)

    return efields_realimag, efields_complex_array, perfect_efields_realimag, perfect_efields_complex_array


def save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list, output_every_iter, pred_c, ni_lists, perfect_efield_list, jac, normalizer=None, debug_list=None, true_exptime_history=None):

    outpath = os.path.dirname(fileout)

    # Initialize lists to collect e-field data across all iterations
    all_efields_complex = []  # Will collect complex e-fields per iteration
    all_perfect_efields_complex = []  # Will collect perfect complex e-fields per iteration

    # Create one subdirectory per iteration
    iters = [len(framelistlist) - 1] if output_every_iter else range(len(framelistlist))
    for i in iters:
        current_debug = debug_list[i] if not output_every_iter and debug_list is not None else None
        current_exptime = true_exptime_history[i] if true_exptime_history is not None else None

        efields_complex_array, perfect_efields_complex_array = save_outputs_iter(
            i, fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list,
            output_every_iter, pred_c, ni_lists, perfect_efield_list[i], jac,
            debugging_dict=current_debug,
            normalizer=normalizer,
            true_exptime_list=current_exptime
        )
        efields_complex_array = np.stack(efields_complex_array, axis=0)
        all_efields_complex.append(efields_complex_array)

        perfect_efields_complex_array = np.stack(perfect_efields_complex_array, axis=0)
        all_perfect_efields_complex.append(perfect_efields_complex_array)

    # If we are outputting every iteration, we have not assembled the correct all_**_efields_complex so need to loop over
    # and properly populate those
    if output_every_iter:
        for i in range(len(framelistlist)):
            efields_realimag, efields_complex_array, perfect_efields_realimag, perfect_efields_complex_array = refactor_efields(
                cfg, otherlist[i], perfect_efield_list=perfect_efield_list[i])

            # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
            efields_complex_array = np.stack(efields_complex_array, axis=0)
            all_efields_complex.append(efields_complex_array)

            perfect_efields_complex_array = np.stack(perfect_efields_complex_array, axis=0)
            all_perfect_efields_complex.append(perfect_efields_complex_array)

    ### Calculate and plot estimation error variance
    # Stack all iterations into final cubes
    efields_datacube = np.stack(all_efields_complex, axis=0)  # (iterations, wavelengths, h, w)
    perfect_efields_datacube = np.stack(all_perfect_efields_complex, axis=0)  # (iterations, wavelengths, h, w)

    # Get dimensions from the first e-field
    nrow, ncol = efields_datacube.shape[2], efields_datacube.shape[3]

    # Create DH mask cube for all wavelengths
    dhmask_cube = []
    for j in range(len(cfg.sl_list)): # We can have more than 3 wavelengths (band 2 and 3)
        dh = cfg.sl_list[j].dh.e
        dhcrop = insertinto(dh, (nrow, ncol)).astype('bool')
        dhmask_cube.append(dhcrop)
        log.info(f"DH mask {j}: {np.sum(dhcrop)} pixels in dark hole out of {dhcrop.size} total")

    # Stack into cube: shape (n_wavelengths, nrow, ncol)
    dhmask_cube = np.stack(dhmask_cube, axis=0)

    # Compute difference cube
    if efields_datacube.shape[1] == perfect_efields_datacube.shape[1]:
        efields_datacube_red = efields_datacube
    else:
        mid_idx = len(cfg.sl_list) // 2
        efields_datacube_red = efields_datacube[:, mid_idx:mid_idx + 1, :, :]

    efield_diff = efields_datacube_red - perfect_efields_datacube

    log.info(f"E-field difference stats:")
    log.info(f"  Shape: {efield_diff.shape}")
    log.info(f"  Min magnitude: {np.min(np.abs(efield_diff))}")
    log.info(f"  Max magnitude: {np.max(np.abs(efield_diff))}")
    log.info(f"  Has NaNs: {np.any(np.isnan(efield_diff))}")
    log.info(f"  Has Infs: {np.any(np.isinf(efield_diff))}")

    # Apply the dh mask and compute variance per wavelength across iterations
    if efields_datacube.shape[0] < 2:
        log.warning(f"Warning: Need at least 2 iterations to compute variance. Current iterations: {efields_datacube.shape[0]}")

    estimation_variance = np.zeros((efield_diff.shape[1], nrow, ncol))  # (n_wavelengths, nrow, ncol)
    variance_per_iter_all_wl = []  # Store variance per iteration for each wavelength

    # Speedup mode or not
    is_speedup = (efield_diff.shape[1] == 1 and len(cfg.sl_list) > 1)
    mid_idx = len(cfg.sl_list) // 2

    for diff_idx in range(efield_diff.shape[1]):  # For each wavelength slice in the residual cube

        mask_idx = mid_idx if is_speedup else diff_idx

        if mask_idx < len(dhmask_cube):
            mask = dhmask_cube[mask_idx]

            # Extract data for this wavelength across all iterations
            wl_diff_data = efield_diff[:, diff_idx, :, :]  # (iterations, nrow, ncol)

            # Apply mask and compute variance across iterations (axis=0)
            masked_diff = wl_diff_data[:, mask]  # (iterations, n_masked_pixels)

            # Compute variance across iterations for each masked pixel
            if masked_diff.size > 0 and efield_diff.shape[0] > 1:
                # For complex data, compute variance of the magnitude
                masked_diff_magnitude = np.abs(masked_diff)  # Convert to magnitude
                pixel_variance = np.nanvar(masked_diff_magnitude, axis=0)  # (n_masked_pixels,) - ignores NaNs

                # Check for problematic values
                nan_count = np.sum(np.isnan(pixel_variance))
                inf_count = np.sum(np.isinf(pixel_variance))
                log.info(f"Wavelength {diff_idx}: {nan_count} NaNs, {inf_count} Infs out of {len(pixel_variance)} pixels")

                # Put the variance values back into the full array
                estimation_variance[diff_idx][mask] = pixel_variance

                # Compute mean variance per iteration for plotting (reuse masked_diff)
                variance_per_iter = []
                for iter_idx in range(efield_diff.shape[0]):
                    iter_data_magnitude = np.abs(masked_diff[iter_idx, :])  # Use magnitude
                    if len(iter_data_magnitude) > 0:
                        variance_per_iter.append(np.nanvar(iter_data_magnitude))  # Ignore NaNs
                    else:
                        variance_per_iter.append(0.0)
                variance_per_iter_all_wl.append(variance_per_iter)
            else:
                log.warning(f"Wavelength {diff_idx}: Insufficient data for variance calculation")
                variance_per_iter_all_wl.append([0.0] * efield_diff.shape[0])
        else:
            variance_per_iter_all_wl.append([0.0] * efield_diff.shape[0])

    # Save estimation variance per pixel to cube
    pyfits.writeto(os.path.join(outpath, "estimation_variance_per_pixel.fits"), estimation_variance, overwrite=True)

    # Save variance per iteration for all wavelengths as CSV table
    if variance_per_iter_all_wl:
        # Convert to numpy array for easier handling
        max_iterations = max(len(wl_data) for wl_data in variance_per_iter_all_wl)
        variance_table = np.zeros((max_iterations, len(variance_per_iter_all_wl)))

        # Fill the table with variance data for each wavelength
        for diff_idx, wl_variance_data in enumerate(variance_per_iter_all_wl):
            variance_table[:len(wl_variance_data), diff_idx] = wl_variance_data

        # Create header with wavelength labels
        header = ','.join([f'Wvln_{diff_idx + 1}' for diff_idx in range(len(variance_per_iter_all_wl))])

        # Save as CSV
        np.savetxt(os.path.join(outpath, "efield_variance.csv"),
                   variance_table, delimiter=",", header=header, comments="")

    # Plot electric field error variance for all wavelengths per iteration
    plt.figure()
    for diff_idx in range(len(variance_per_iter_all_wl)):  # Plot all wavelengths
        variance_per_iter = variance_per_iter_all_wl[diff_idx]
        plt.plot(np.arange(len(variance_per_iter)) + 1, variance_per_iter,
                marker='o', label=f'Wavelength {diff_idx + 1}')

    plt.xlabel('Iteration')
    plt.ylabel('Electric Field Variance')
    plt.semilogy()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, len(variance_per_iter) + 1))
    plt.savefig(os.path.join(outpath, "efield_variance.pdf"))
    plt.close()


def save_debugging_iteration(debugging_dict, iteration, outpath,
                              csv_path='debugging_history.csv'):
    """
    Append per-wavelength scalar quantities to a CSV (one row per wavelength per iteration).

    Parameters
    ----------
    debugging_dict : dict
        The debugging dictionary returned each iteration.
    iteration : int
        Current iteration index.
    csv_path : str
        Path to the output CSV file (appended each call).
    """
    nlam = debugging_dict['peakflux'].shape[0]

    fieldnames = [
        'iteration', 'lam_index', 'beta',
        'peakflux', 'next_c', 'this_iter_dur', 'this_iter_dur_gitl',
        'cam_nom_gain',    'cam_nom_exptime',    'cam_nom_nframes',
        'cam_probe_gain',  'cam_probe_exptime',  'cam_probe_nframes',
        'pred_mean_contrast',         'pred_bright_contrast',
        'pred_mean_contrast_probing', 'pred_bright_contrast_probing',
    ]

    debugging_csv_path = os.path.join(outpath, csv_path)
    write_header = not os.path.exists(debugging_csv_path)
    with open(debugging_csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for j in range(nlam):
            writer.writerow({
                'iteration': iteration,
                'lam_index': j,
                'beta': debugging_dict['beta'],
                'peakflux':  debugging_dict['peakflux'][j, 0],
                'next_c':    debugging_dict['next_c'],
                'this_iter_dur': debugging_dict['this_iter_time'],
                'this_iter_dur_gitl': debugging_dict['this_iter_time'] + 60*60,
                'cam_nom_gain':      debugging_dict['cam_params']['nom'][j, 0],
                'cam_nom_exptime':   debugging_dict['cam_params']['nom'][j, 1],
                'cam_nom_nframes':   debugging_dict['cam_params']['nom'][j, 2],
                'cam_probe_gain':    debugging_dict['cam_params']['probing'][j, 0],
                'cam_probe_exptime': debugging_dict['cam_params']['probing'][j, 1],
                'cam_probe_nframes': debugging_dict['cam_params']['probing'][j, 2],
                'pred_mean_contrast':
                    debugging_dict['cam_params_inputs']['pred_mean_contrast'][j, 0],
                'pred_bright_contrast':
                    debugging_dict['cam_params_inputs']['pred_bright_contrast'][j, 0],
                'pred_mean_contrast_probing':
                    debugging_dict['cam_params_inputs']['pred_mean_contrast_probing'][j, 0],
                'pred_bright_contrast_probing':
                    debugging_dict['cam_params_inputs']['pred_bright_contrast_probing'][j, 0],
            })


def save_normalized_images_cube(iterpath, flist, true_exptime_list, debugging_dict, nlam, hdr, cam_params_hdu, normalizer):
    """
    Converts the images.fits cube in contrast unit.

    Iterates through each frame of the raw sequence (flist) which contains images probed and unprobed, extracts the specific
    exposure time strictly associated with that frame, and applies the normalization.

    Parameters
    ----------
    iterpath : str
        Absolute path to the current iteration output directory.
    flist : list of np.ndarray
        List of 2D raw detector images. Total size must be nlam * nframes_per_lam.
    true_exptime_list : list or 1D array, optional
        List of exact exposure times corresponding to each frame in `flist` for the
        current iteration. Used for the normalization of images in contrast unit, bypassing
        the offset `camlist`. Defaults to None.
    debugging_dict : dict
        Dictionary from the main loop containing 'peakflux' per wavelength (necessary for the conversion).
    nlam : int
        Number of spectral channels.
    hdr : astropy.io.fits.Header
        FITS header to attach to the primary HDU.
    cam_params_hdu : astropy.io.fits.ImageHDU
        Camera parameters HDU to append for scientific traceability.
    normalizer : object
        The instantiated normalization object handling
        the specific noise regime logic and scaling factors.

    Returns
    -------
    None
    """
    if normalizer is None or debugging_dict is None:
        log.warning("Missing normalization tools. Bypassing images_contrast.fits generation.")
        return

    nframes = len(flist)
    nframes_per_lam = nframes // nlam
    normalized_flist = []

    img_ctr = pyfits.ImageHDU(name='NORMALIZED_IMAGES')

    for idx, raw_im in enumerate(flist):
        lam_idx = idx // nframes_per_lam

        try:
            peakflux = float(debugging_dict['peakflux'][lam_idx, 0])
            exptime = float(true_exptime_list[idx])
        except IndexError:
            log.error(f"Index {idx} out of bounds. Cannot normalize frame.")
            return

        # Normalise to contrast units the images
        norm_im = normalizer.normalize(raw_im, peakflux, exptime)
        normalized_flist.append(norm_im)

        img_ctr.header[f'EXP{idx:03d}'] = (exptime, f'Exptime (s) frame {idx}')
        img_ctr.header[f'PFL{idx:03d}'] = (peakflux, f'Peakflux frame {idx}')
        img_ctr.header[f'LAM{idx:03d}'] = (lam_idx, f'Lambda index frame {idx}')

    # Saving the data in the iteration folder
    img_ctr.data = np.array(normalized_flist)
    prim = pyfits.PrimaryHDU(header=hdr)
    hdul_ctr = pyfits.HDUList([prim, img_ctr, cam_params_hdu])

    out_file = os.path.join(iterpath, "images_contrast.fits")
    hdul_ctr.writeto(out_file, overwrite=True)
    log.info(f"Normalized contrast cube saved to {out_file}")