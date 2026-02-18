import os
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np

from howfsc.util.gitl_tools import param_order_to_list
from howfsc.util.insertinto import insertinto

def save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list):

    outpath = os.path.dirname(fileout)

    # Plot measured_c vs iteration
    plt.figure()
    plt.plot(np.arange(len(measured_c)) + 1, measured_c, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Measured Contrast')
    plt.semilogy()
    plt.xticks(np.arange(1, len(measured_c) + 1))
    plt.savefig(os.path.join(outpath, "contrast_vs_iteration.pdf"), bbox_inches='tight')
    plt.close()

    # Save measured_c to a csv file
    np.savetxt(os.path.join(outpath, "measured_contrast.csv"), np.array(measured_c), delimiter=",",
               header="Measured Contrast", comments="")

    # Create one subdirectory per iteration
    for i in range(len(framelistlist)):
        iterpath = os.path.join(outpath, f"iteration_{i + 1:04d}")
        if not os.path.exists(iterpath):
            os.makedirs(iterpath)

    # Initialize lists to collect e-field data across all iterations
    all_efields_complex = []  # Will collect complex e-fields per iteration
    all_perfect_efields_complex = []  # Will collect perfect complex e-fields per iteration

    # Saving separate intensity files per iteration
    for i, flist in enumerate(framelistlist):
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
        efields = []
        efields_complex = []
        for n in range(len(cfg.sl_list)):
            efields.append(np.real(oitem[n]['meas_efield']))
            efields.append(np.imag(oitem[n]['meas_efield']))
            efields_complex.append(oitem[n]['meas_efield'])

        # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
        efields_complex_array = np.stack(efields_complex, axis=0)
        all_efields_complex.append(efields_complex_array)

        hdr_ef = pyfits.Header()
        hdr_ef['NLAM'] = len(cfg.sl_list)
        prim_ef = pyfits.PrimaryHDU(header=hdr_ef)
        img_ef = pyfits.ImageHDU(np.array(efields))
        hdul_ef = pyfits.HDUList([prim_ef, img_ef])
        hdul_ef.writeto(os.path.join(iterpath, "efield_estimations.fits"), overwrite=True)

        # --- PERFECT E-FIELDS ---
        perfect_efields_list = []
        perfect_efields_complex = []
        for n in range(len(cfg.sl_list)):
            perfect_efields_list.append(np.real(oitem[n]['model_efield']))
            perfect_efields_list.append(np.imag(oitem[n]['model_efield']))
            perfect_efields_complex.append(oitem[n]['model_efield'])

        # Convert to numpy array for this iteration: shape (n_wavelengths, height, width)
        perfect_efields_complex_array = np.stack(perfect_efields_complex, axis=0)
        all_perfect_efields_complex.append(perfect_efields_complex_array)

        hdr_pef = pyfits.Header()
        hdr_pef['NLAM'] = len(cfg.sl_list)
        prim_pef = pyfits.PrimaryHDU(header=hdr_pef)
        img_pef = pyfits.ImageHDU(np.array(perfect_efields_list))
        hdul_pef = pyfits.HDUList([prim_pef, img_pef])
        hdul_pef.writeto(os.path.join(iterpath, "perfect_efields.fits"), overwrite=True)

        # --- DM STATES (PER ITERATION) ---
        if dm1_list is not None and i < len(dm1_list):
            pyfits.writeto(os.path.join(iterpath, "dm1_command.fits"), dm1_list[i], overwrite=True)

        if dm2_list is not None and i < len(dm2_list):
            pyfits.writeto(os.path.join(iterpath, "dm2_command.fits"), dm2_list[i], overwrite=True)

        print(f"Saved outputs (individual) for iteration {i + 1}")

    ### Calculate and plot estimation error variance
    # Stack all iterations into final cubes
    efields_datacube = np.stack(all_efields_complex, axis=0)  # (iterations, wavelengths, h, w)
    perfect_efields_datacube = np.stack(all_perfect_efields_complex, axis=0)  # (iterations, wavelengths, h, w)

    # Get dimensions from the first e-field
    nrow, ncol = efields_datacube.shape[2], efields_datacube.shape[3]

    # Create DH mask cube for all three wavelengths (indices 0-2)
    dhmask_cube = []
    for j in range(min(3, len(cfg.sl_list))):  # Use up to 3 wavelengths or fewer if available
        dh = cfg.sl_list[j].dh.e
        dhcrop = insertinto(dh, (nrow, ncol)).astype('bool')
        dhmask_cube.append(dhcrop)
        print(f"DH mask {j}: {np.sum(dhcrop)} pixels in dark hole out of {dhcrop.size} total")

    # Stack into cube: shape (n_wavelengths, nrow, ncol)
    dhmask_cube = np.stack(dhmask_cube, axis=0)

    # Compute difference cube
    efield_diff = efields_datacube - perfect_efields_datacube

    print(f"E-field difference stats:")
    print(f"  Shape: {efield_diff.shape}")
    print(f"  Min magnitude: {np.min(np.abs(efield_diff))}")
    print(f"  Max magnitude: {np.max(np.abs(efield_diff))}")
    print(f"  Has NaNs: {np.any(np.isnan(efield_diff))}")
    print(f"  Has Infs: {np.any(np.isinf(efield_diff))}")

    # Apply the dh mask and compute variance per wavelength across iterations
    if efields_datacube.shape[0] < 2:
        print("Warning: Need at least 2 iterations to compute variance. Current iterations:", efields_datacube.shape[0])

    estimation_variance = np.zeros((efield_diff.shape[1], nrow, ncol))  # (n_wavelengths, nrow, ncol)
    variance_per_iter_all_wl = []  # Store variance per iteration for each wavelength

    for wl_idx in range(efield_diff.shape[1]):  # For each wavelength
        if wl_idx < len(dhmask_cube):  # Only if we have a mask for this wavelength
            # Get the mask for this wavelength
            mask = dhmask_cube[wl_idx]

            # Extract data for this wavelength across all iterations
            wl_diff_data = efield_diff[:, wl_idx, :, :]  # (iterations, nrow, ncol)

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
                print(f"Wavelength {wl_idx}: {nan_count} NaNs, {inf_count} Infs out of {len(pixel_variance)} pixels")

                # Put the variance values back into the full array
                estimation_variance[wl_idx][mask] = pixel_variance

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
                print(f"Wavelength {wl_idx}: Insufficient data for variance calculation")
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
        for wl_idx, wl_variance_data in enumerate(variance_per_iter_all_wl):
            variance_table[:len(wl_variance_data), wl_idx] = wl_variance_data

        # Create header with wavelength labels
        header = ','.join([f'Wvln_{wl_idx + 1}' for wl_idx in range(len(variance_per_iter_all_wl))])

        # Save as CSV
        np.savetxt(os.path.join(outpath, "efield_variance.csv"),
                   variance_table, delimiter=",", header=header, comments="")

    # Plot electric field error variance for all wavelengths per iteration
    plt.figure()
    for wl_idx in range(min(3, len(variance_per_iter_all_wl))):  # Plot up to 3 wavelengths
        variance_per_iter = variance_per_iter_all_wl[wl_idx]
        plt.plot(np.arange(len(variance_per_iter)) + 1, variance_per_iter,
                marker='o', label=f'Wavelength {wl_idx + 1}')

    plt.xlabel('Iteration')
    plt.ylabel('Electric Field Variance')
    plt.semilogy()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, len(variance_per_iter) + 1))
    plt.savefig(os.path.join(outpath, "efield_variance.pdf"))
    plt.close()
