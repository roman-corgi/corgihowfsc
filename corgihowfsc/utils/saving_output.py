import os
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np

from howfsc.util.gitl_tools import param_order_to_list

def save_outputs(fileout, cfg, camlist, framelistlist, otherlist, measured_c, dm1_list, dm2_list):

    outpath = os.path.dirname(fileout)

    # Plot measured_c vs iteration
    plt.figure()
    plt.plot(np.arange(len(measured_c)) + 1, measured_c, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Measured Contrast')
    plt.semilogy()
    plt.savefig(os.path.join(outpath, "contrast_vs_iteration.pdf"))
    plt.close()

    # Save measured_c to a csv file
    np.savetxt(os.path.join(outpath, "measured_contrast.csv"), np.array(measured_c), delimiter=",",
               header="Measured Contrast", comments="")

    # Create one subdirectory per iteration
    for i in range(len(framelistlist)):
        iterpath = os.path.join(outpath, f"iteration_{i + 1:04d}")
        if not os.path.exists(iterpath):
            os.makedirs(iterpath)

    # Saving separate intensity files per iteration
    for i, flist in enumerate(framelistlist):
        oitem = otherlist[i]

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
        hdul_main.writeto(os.path.join(outpath, f"iteration_{i + 1:04d}", "images.fits"), overwrite=True)

        # Saving total intensity
        prim_tot = pyfits.PrimaryHDU(header=hdr)
        img_tot = pyfits.ImageHDU(np.array(stack_total), name='TOTAL_INTENSITY')
        hdul_tot = pyfits.HDUList([prim_tot, img_tot])
        hdul_tot.writeto(os.path.join(outpath, f"iteration_{i + 1:04d}", "intensity_total.fits"), overwrite=True)

        # Saving coherent intensity
        prim_coh = pyfits.PrimaryHDU(header=hdr)
        img_coh = pyfits.ImageHDU(np.array(stack_coh), name='COHERENT_INTENSITY')
        hdul_coh = pyfits.HDUList([prim_coh, img_coh])
        hdul_coh.writeto(os.path.join(outpath, f"iteration_{i + 1:04d}", "intensity_coherent.fits"), overwrite=True)

        # Saving incoherent intensity
        prim_incoh = pyfits.PrimaryHDU(header=hdr)
        img_incoh = pyfits.ImageHDU(np.array(stack_incoh), name='INCOHERENT_INTENSITY')
        hdul_incoh = pyfits.HDUList([prim_incoh, img_incoh])
        hdul_incoh.writeto(os.path.join(outpath, f"iteration_{i + 1:04d}", "intensity_incoherent.fits"), overwrite=True)

        # --- E-FIELD ESTIMATIONS ---
        efields = []
        for n in range(len(cfg.sl_list)):
            efields.append(np.real(oitem[n]['meas_efield']))
            efields.append(np.imag(oitem[n]['meas_efield']))

        hdr_ef = pyfits.Header()
        hdr_ef['NLAM'] = len(cfg.sl_list)
        prim_ef = pyfits.PrimaryHDU(header=hdr_ef)
        img_ef = pyfits.ImageHDU(efields)
        hdul_ef = pyfits.HDUList([prim_ef, img_ef])
        hdul_ef.writeto(os.path.join(outpath, f"iteration_{i + 1:04d}", "efield_estimations.fits"), overwrite=True)

        print(f"Saved outputs (individual) for iteration {i + 1}")


    # Check that DM lists are present
    if dm1_list is not None and dm2_list is not None:

        # Create 'dm_data' subdirectory in the output folder
        dm_outpath = os.path.join(outpath, "dm_data")
        if not os.path.exists(dm_outpath):
            os.makedirs(dm_outpath)

        # Save final maps (for re-seeding) : take the last element of the lists
        if len(dm1_list) > 0:
            pyfits.writeto(os.path.join(dm_outpath, "final_dm1.fits"), dm1_list[-1], overwrite=True)
        if len(dm2_list) > 0:
            pyfits.writeto(os.path.join(dm_outpath, "final_dm2.fits"), dm2_list[-1], overwrite=True)

        # Save history cubes (full commands)
        if len(dm1_list) > 0:
            dm1_cube = np.array(dm1_list)
            hdr_dm = pyfits.Header()
            hdr_dm['CONTENT'] = 'DM1 VOLTAGE HISTORY'
            hdr_dm['UNIT'] = 'Volts'
            pyfits.writeto(os.path.join(dm_outpath, "dm1_command_history.fits"),
                           dm1_cube, header=hdr_dm, overwrite=True)

        if len(dm2_list) > 0:
            dm2_cube = np.array(dm2_list)
            hdr_dm = pyfits.Header()
            hdr_dm['CONTENT'] = 'DM2 VOLTAGE HISTORY'
            hdr_dm['UNIT'] = 'Volts'
            pyfits.writeto(os.path.join(dm_outpath, "dm2_command_history.fits"),
                           dm2_cube, header=hdr_dm, overwrite=True)

        print(f"Global cubes and final DM maps saved to {dm_outpath}")