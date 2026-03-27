import os
import csv
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np

from howfsc.util.gitl_tools import param_order_to_list
from howfsc.util.insertinto import insertinto

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

def save_outputs_iter(i, fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list, output_every_iter, pred_c, ni_lists, perfect_efield_list, debugging_dict=None):

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

    hdr = pyfits.Header()
    hdr['NLAM'] = len(cfg.sl_list)
    hdr['ITER'] = i + 1

    # Saving data of images
    prim = pyfits.PrimaryHDU(header=hdr)
    img_raw = pyfits.ImageHDU(flist, name='RAW_IMAGES')
    prev = pyfits.ImageHDU(param_order_to_list(camlist[i][1]), name='CAM_PARAMS')

    hdul_main = pyfits.HDUList([prim, img_raw, prev])
    hdul_main.writeto(os.path.join(iterpath, "images.fits"), overwrite=True)

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


def save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list, output_every_iter, pred_c, ni_lists, perfect_efield_list):

    outpath = os.path.dirname(fileout)

    # Initialize lists to collect e-field data across all iterations
    all_efields_complex = []  # Will collect complex e-fields per iteration
    all_perfect_efields_complex = []  # Will collect perfect complex e-fields per iteration

    # Create one subdirectory per iteration
    iters = [len(framelistlist)-1] if output_every_iter else range(len(framelistlist))
    for i in iters:
        efields_complex_array, perfect_efields_complex_array = save_outputs_iter(i, fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list, output_every_iter, pred_c, ni_lists, perfect_efield_list[i])
        # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
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
        'peakflux', 'next_c',
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

