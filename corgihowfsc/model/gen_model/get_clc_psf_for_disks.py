# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test accuracy of SPAM calibration."""
import os
import pathlib
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import pandas as pd

import cgisim as cgisim
import proper


from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample
from cal.pupilfit import pupilfit_gsw
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml

PUPILFIT_PATH = pathlib.Path(pupilfit_gsw.__file__).resolve().parent

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE.split('model')[0], 'model')
GEN_MODEL_PATH = os.path.join(MODEL_PATH, 'gen_model')
IN_PATH = os.path.join(GEN_MODEL_PATH, 'in')
OUT_PATH = os.path.join(GEN_MODEL_PATH, 'out')
MASK_PATH = os.path.join(IN_PATH, 'mask_designs')
DATA_PATH_PUPIL = os.path.join(MASK_PATH, 'pupil')
DM1_PATH = os.path.join(MODEL_PATH, 'dm1')
DM2_PATH = os.path.join(MODEL_PATH, 'dm2')


def plot_contrast():

    img_hlc = fits.getdata('/Users/ajriggs/Downloads/psf_clc_hlc_fpm_hlc_ls.fits')
    hlc_unmasked = fits.getdata('/Users/ajriggs/Downloads/psf_clc_no_fpm_hlc_ls.fits')

    img_wfov = fits.getdata('/Users/ajriggs/Downloads/psf_clc_hlc_fpm_wfov_ls.fits')
    wfov_unmasked = fits.getdata('/Users/ajriggs/Downloads/psf_clc_no_fpm_wfov_ls.fits')


    vmin=-7
    vmax=-2

    plt.figure()
    plt.imshow(np.log10(img_hlc/np.max(hlc_unmasked)), cmap='magma', vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig('/Users/ajriggs/Downloads/img_masked_hlc_ls.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure()
    plt.imshow(np.log10(img_wfov/np.max(wfov_unmasked)), cmap='magma', vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig('/Users/ajriggs/Downloads/img_masked_wfov_ls.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    




def plot_lyot():

    n = 321
    imshape = (n, n)
    pupil = inin(fits.getdata('pupil.fits'), imshape)
    lyot_wfov = inin(fits.getdata('wfov_lyot_stop_309.fits'), imshape)
    lyot_hlc = inin(fits.getdata('lyot_rotated.fits'), imshape)

    plt.figure()
    plt.imshow(pupil, cmap='gray')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/pupil.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure()
    plt.imshow(lyot_wfov, cmap='gray')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/lyot_wfov.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure()
    plt.imshow(lyot_hlc, cmap='gray')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/lyot_hlc.png', bbox_inches='tight', pad_inches=0.1, dpi=300)



    plt.figure()
    plt.imshow(lyot_wfov + pupil, cmap='magma')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/lyot_superimposed_wfov.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    plt.figure()
    plt.imshow(lyot_hlc + pupil, cmap='magma')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/lyot_superimposed_hlc.png', bbox_inches='tight', pad_inches=0.1, dpi=300)


def plot_psf_core():

    n = 15
    psf_shape = (n, n)
    # psf_shape = (21, 21)

    psf_no_ls = inin(fits.getdata('/Users/ajriggs/Downloads/psf_no_masks_hlc.fits'), psf_shape)
    psf_wfov_ls = inin(fits.getdata('/Users/ajriggs/Downloads/psf_clc_no_fpm_wfov_ls.fits'), psf_shape)
    psf_hlc_ls = inin(fits.getdata('/Users/ajriggs/Downloads/psf_clc_no_fpm_hlc_ls.fits'), psf_shape)

    for psf in [psf_no_ls,  psf_wfov_ls, psf_hlc_ls,]:
        maxval = np.max(psf)
        halfmaxsum = np.sum(psf[psf > maxval/2])
        print('Half max sum = %.2e' % halfmaxsum)



    maxval = np.max(psf_no_ls)
    plt.figure(1)
    plt.plot(np.arange(n), psf_no_ls[n//2, :]/maxval, label='No Lyot stop')
    plt.plot(np.arange(n), psf_wfov_ls[n//2, :]/maxval, label='WFOV Lyot stop')
    plt.plot(np.arange(n), psf_hlc_ls[n//2, :]/maxval, label='HLC Lyot stop')
    plt.xlabel('pixels')
    plt.legend()
    plt.savefig('/Users/ajriggs/Downloads/psf_core_cross_sections.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure()
    plt.imshow(psf_wfov_ls)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/psf_wfov_ls.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    plt.figure()
    plt.imshow(psf_hlc_ls)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig('/Users/ajriggs/Downloads/psf_hlc_ls.png', bbox_inches='tight', pad_inches=0.1, dpi=300)


    plt.show()


    pass    



def gen_psf():
    """Generate PSFs."""
    flagPlot = True

    cgi_mode = 'excam' #'excam_efield'
    # imshape = (351, 351)
    polaxis = -10  # compute images for mean X+Y polarization
    # use_errors = True
    # star_spectrum = 'a0v'  # 'k5v'
    # star_vmag = 2.0

    v_dm1 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm1_v.fits' ))
    v_dm2 = proper.prop_fits_read(os.path.join(IN_PATH, 'flatmaps', 'hlc_flat_wfe_dm2_v.fits' ))

    # ccd_fn = os.path.join(IN_PATH, 'ccd_params_for_pupil_imaging.yaml')
    # ccd = loadyaml(ccd_fn)
    # ccd.update({'exptime': 0.8})
    ccd = {}

    # magVec = [1, ]  # np.linspace(0.95, 1.05, 6)
    # rotVec = [0, ]  # np.arange(-2, 3)

    bandList = ['1']  # ['1b', '2c', '3c', '4b']
    rotTrueList = []
    magTrueList = []
    rotEstList = []
    magEstList = []
    diamList = []

    magTrue = 1
    rotTrue = 0

    for bandpass in bandList:


        # %% Unmasked PSF, HLC
        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            # 'use_pupil_mask': 0,  # shaped pupil
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':0,
            'use_lyot_stop':0,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'hlc_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_no_masks_hlc.fits', image0, overwrite=True)


        # %% CLC PSF (HLC)
        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            # 'use_pupil_mask': 0,  # shaped pupil
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':1,
            'use_lyot_stop':1,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'hlc_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_clc_hlc_fpm_hlc_ls.fits', image0, overwrite=True)



        # %% CLC PSF with HLC LS only (HLC)
        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            # 'use_pupil_mask': 0,  # shaped pupil
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':0,
            'use_lyot_stop':1,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'hlc_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_clc_no_fpm_hlc_ls.fits', image0, overwrite=True)



        # %% CLC PSF with SPC LS (HLC)
        lyot0 = fits.getdata('LS_SPC-20200610_1000.fits')
        mag = 309/1000
        lyot_stop_array = downsample(lyot0, 0, mag, 0, 0)
        fits.writeto('/Users/ajriggs/Downloads/wfov_lyot_stop_309.fits', lyot_stop_array, overwrite=True)

        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            # 'use_pupil_mask': 0,  # shaped pupil
            'lyot_stop_array': lyot_stop_array,
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':1,
            'use_lyot_stop':1,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'hlc_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_clc_hlc_fpm_wfov_ls.fits', image0, overwrite=True)


        # %% CLC PSF with SPC LS only (HLC)
        lyot0 = fits.getdata('LS_SPC-20200610_1000.fits')
        mag = 309/1000
        lyot_stop_array = downsample(lyot0, 0, mag, 0, 0)

        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            # 'use_pupil_mask': 0,  # shaped pupil
            'lyot_stop_array': lyot_stop_array,
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':0,
            'use_lyot_stop':1,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'hlc_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_clc_no_fpm_wfov_ls.fits', image0, overwrite=True)


        # %% Unmasked PSF, SPC
        print("Computing PSF")
        params = {
            # 'pupil_mask_array': pupil_mask_array,
            'use_pupil_mask': 0,  # shaped pupil
            'ccd':{},
            'use_pupil_lens':0,
            'use_errors':1,
            'use_fpm':0,
            'use_lyot_stop':0,
            'use_field_stop':0,
            'use_dm1':1,
            'dm1_v':v_dm1,
            'use_dm2':1,
            'dm2_v':v_dm2,
            }

        cor_type = 'spc-wide_band1'
        image0, counts = cgisim.rcgisim(
            cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
            # star_spectrum=star_spectrum, star_vmag=star_vmag,
        )
        imageNorm = image0/np.max(image0)

        fits.writeto('/Users/ajriggs/Downloads/psf_no_masks_spc.fits', image0, overwrite=True)







        # 


        # field_cube, _ = cgisim.rcgisim(
        #     cgi_mode, cor_type, bandpass, polaxis, params, ccd=ccd,
        #     # star_spectrum=star_spectrum, star_vmag=star_vmag,
        # )
        # field = np.mean(field_cube, axis=0)
        # field = insertinto(field, imshape)
        # amp = np.abs(field)
        # image0 = amp**2
        # image = image0/np.max(image0)
        # image -= ccd['bias']
        # image = np.rot90(image, -1)  # 90 deg CCW rot to be in EXCAM coords
        # image = insertinto(image, (gridsize, gridsize))
        # image = np.roll(image, (yOffset, xOffset), axis=(0, 1))
        # image[image < 0] = 0

        if flagPlot:
            plt.figure(1)
            plt.clf()
            plt.title(f'Band {bandpass.upper()}')
            plt.imshow(imageNorm)
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.pause(1)

            plt.show()


    #     # ######################################################
    #     # Test that the unmasked pupil parameters are fitted correctly.
    #     fn_tuning = os.path.join(OUT_PATH, 'spam', 'params_fit_spam_mag_clocking_band%s.yaml' % bandpass.lower())
    #     # fnYAML = os.path.join(IN_PATH, 'any_band', 'params_fit_spam_mag_clocking.yaml')
    #     # fnYAML = os.path.join(IN_PATH, 'any_band', 'fit_spam_mag_clocking.yaml')
    #     magEst, rotEst = pupilfit_gsw.fit_shaped_pupil_mag_clocking(image, fn_tuning)
    #     print('rotEst = %.3f  magEst = %.4f\n' % (rotEst, magEst))

    #     tuning_dict = loadyaml(fn_tuning)
    #     diamList.append(tuning_dict['nBeamNom'])
    #     # nBeamNom = tuning_dict['nBeamNom']
    #     # nBeam = nBeamNom*magEst
    #     # diamList.append(nBeam)
    #     rotEstList.append(rotEst)
    #     magEstList.append(magEst)

    # data = {
    #     'band': bandList,
    #     'beamDiamPix': diamList,
    #     'rotTrue (degrees)': rotTrueList,
    #     'rotEst (degrees)': rotEstList,
    #     'rotError (degrees)':
    #         np.array(rotTrueList) - np.array(rotEstList),
    #     'magTrue': magTrueList,
    #     'magEst': magEstList,
    #     'magError (%)': (np.array(magTrueList) - np.array(magEstList))*100,
    # }

    # df = pd.DataFrame(data=data)  # , index=row_labels)

    # out_name = os.path.join(OUT_PATH, 'spam', 'calib_spam_results.csv')
    # df.to_csv(out_name)


if __name__ == '__main__':
    # gen_psf()
    # plot_psf_core()
    plot_lyot()
    # plot_contrast()
