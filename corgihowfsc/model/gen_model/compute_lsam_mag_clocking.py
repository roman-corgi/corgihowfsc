"""
Compute the clocking and magnification of all the Lyot stops using images with
and without the SPC Spec Lyot stop in place.

Example usage:
    python compute_lsam_mag_clocking.py <id_unmasked> <id_masked> 1


"""
import argparse
import os
import sys

from astropy.io import fits
import matplotlib.pyplot as plt

from cal.pupilfit.pupilfit_gsw import fit_lyot_stop_mag_clocking
import cal.pupilfit.pupilfit_open as pfo
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml

import __main__ as main

# init dorba
import ctc.dorba
ctc.dorba.init(data_subdir=os.environ['USER'])

HERE = os.path.dirname(os.path.abspath(__file__))
CGI_AUX_PATH = os.path.join(HERE.split('cgi-orbit')[0], 'cgi-aux')
CGI_LDS_PATH = os.path.join(HERE.split('cgi-orbit')[0], 'cgi-data-structures')
ORBIT_PATH = os.path.join(HERE.split('cgi-orbit')[0], 'cgi-orbit')
DB_PATH = os.path.join(ORBIT_PATH, 'database')
UTIL_PATH = os.path.join(ORBIT_PATH, 'util')
MODEL_PATH = os.path.join(ORBIT_PATH, 'model')
YAML_PATH = os.path.join(MODEL_PATH, 'any_band')
MASK_PATH = os.path.join(MODEL_PATH, 'mask_designs')
TEST_DATA_PATH = os.path.join(UTIL_PATH, 'testdata')

# append necessary cgi-orbit paths for importing
sys.path.append(DB_PATH)
sys.path.append(UTIL_PATH)
from filename import abs_to_dorba, extract_first_number
from get_frames_by_id import get_frames_by_id

# FROZEN
FN_MASK_PARAMS = os.path.join(YAML_PATH, 'params_lyot_stop_mask_def_spec.yaml')
DATA_PATH_SPEC = os.path.join(MASK_PATH, 'SPC_20200617_Spec') 
WAVELENGTH_LUT = loadyaml(path=abs_to_dorba(
    os.path.join(MODEL_PATH, 'misc', 'subband_center_wavelengths.yaml')))
# Defaults
BAND_CHOICES = ['1B', '2B', '3C', '4B']
FN_PUPILFIT_TUNING = os.path.join(YAML_PATH, 'params_fit_unmasked_pupil.yaml')
FN_OFFSET_PARAMS_DEFAULT = os.path.join(YAML_PATH, 'params_fit_pupil_mask_offsets.yaml')
FN_LYOT_CALIB_DEFAULT = os.path.join(YAML_PATH, 'params_fit_lsam_mag_clocking.yaml')
DATA_PATH_PUPILFIT_DEFAULT = os.path.join(MASK_PATH, 'pupil')
# BAND_NUM_CHOICES = (1, 2, 3, 4)


def compute_lsam_mag_clocking(
        frame_id_unmasked,
        frame_id_masked,
        subband_name,
        write=True,
        fn_lyot_calib=FN_LYOT_CALIB_DEFAULT,
        fn_offset_params=FN_OFFSET_PARAMS_DEFAULT,
        data_path_pupilfit=DATA_PATH_PUPILFIT_DEFAULT,
        show_plot=True,
        ):
    """
    Estimate rotation (clocking) and maginification of Lyot mask relative to the
    telescope pupil and the model grid. Apply the estimated rotation to the measured
    relative rotation of the other Lyot masks.

    Parameters:
      framd_id_unmasked : int, EDS frame ID for the EXCAM pupil image with masks open

      frame_id_masked : str, EDS frame ID for the EXCAM pupil image with SPC_SPEC Lyot mask in

      subband_name : str, Which color filter the images were taken with.

      write : bool, optional, Wether to write the results to a YAML file.

      fn_lyot_calib : str, optional, full filename for yaml parameters file passed to
        fit_lyot_stop_mag_clocking()

      fn_offset_params : str, optional, full filename for yaml parameters file passed to
        fit_lyot_stop_mag_clocking()

      data_path_pupilfit : str, optional, full filename for yaml parameters file passed to
        fit_lyot_stop_mag_clocking()

      show_plot : bool, optional, if True display the EXCAM images and the Lyot mask model

    """
    subband_name = subband_name.lower()
    band_num = extract_first_number(subband_name)
    lam = WAVELENGTH_LUT[subband_name]

    frame_list, hdr_list = get_frames_by_id([frame_id_unmasked, ])
    pupil_img = frame_list[0]

    frame_list, hdr_list = get_frames_by_id([frame_id_masked, ])
    masked_img = frame_list[0]

    # 1. Fit the unmasked pupil image to return the x- and y-offsets and the unmasked image as a 2-D array.
    xOffset_pupil, yOffset_pupil, clockEst_pupil, diamEst_pupil = \
        pfo.fit_unmasked_pupil(
            pupil=pupil_img,
            fn_tuning=abs_to_dorba(FN_PUPILFIT_TUNING),
            data_path=abs_to_dorba(data_path_pupilfit),
            )

    # 2. Run cal.pupilfit.pupilfit_fit_lyot_stop_mag_clocking
    #    return the magnification relative to the estimated pupil diameter and
    #    the clocking of the SPC Spec Lyot stop.
    # function_name = 'cal.pupilfit.pupilfit_fit_lyot_stop_mag_clocking'
    inputs = {
        'imageUnmasked': pupil_img,
        'imageMasked': masked_img,
        'xOffsetPupil': xOffset_pupil,
        'yOffsetPupil': yOffset_pupil,
        'fnMaskParams': abs_to_dorba(FN_MASK_PARAMS),
        'fnOffsetParams': abs_to_dorba(fn_offset_params),
        'fnLyotCalib': abs_to_dorba(fn_lyot_calib),
        'data_path': abs_to_dorba(DATA_PATH_SPEC),
    }

    try:

        magEst_mask, clockEst_mask = fit_lyot_stop_mag_clocking(**inputs)
        # magEst_mask = estimated magnification of Lyot stop to nominal telescope pupil (at LSAM)
        # clockEst_mask (deg) = clocking of Lyot stop relative to reference Lyot stop file

    except Exception as ex:
        # docker call failed
        print(ex)

        # force show_plot for sanity check
        show_plot = True

    # 3. Add the known, hard-coded delta clockings between Lyot stops
    #    (as pulled from the spreadsheet on the CGI Wiki)
    #    to the SPC Lyot stop to compute the clockings of the other 3 Lyot stops.
    # LSAM_table = pandas.read_csv(os.path.join(HERE, 'csv', 'LSAM_Mask_Positions_10492306-1.csv'))
    # Note: measured delta clockings are < 0.02 deg, and so are ignored
    #    HLC NFOV Lyot stop clocking in degrees
    #    SPC WFOV Lyot stop clocking in degrees
    #    SPC Spec Rot Lyot stop clocking in degrees
    clockEst_Lyot_HLC_NFOV = clockEst_mask
    clockEst_Lyot_SPC_WFOV = clockEst_mask
    clockEst_Lyot_SPC_SPEC_Rot = clockEst_mask

    # 4. Compute the magnification factor relative to the 1000x1000 reference file.
    #    This involves dividing out the pupil diameter from the YAML config file,
    #    NOT the one just computed by the script fit_unmasked_pupil.py.
    #
    # If you get a magnification of say 0.98 (relative to the beam diameter),
    # then you have to scale by diamPupil/diamHighResMaskRef,
    # where diamPupil is from:
    # cgi-orbit/model/any_band/params_fit_lsam_mag_clocking.yaml
    # and diamHighResMaskRef is from:
    # cgi-orbit/model/any_band/params_lyot_stop_mask_def_nfov.yaml

    params_lyot = loadyaml(path=abs_to_dorba(fn_lyot_calib))
    params_mask = loadyaml(path=abs_to_dorba(FN_MASK_PARAMS))
    mag_lyot_model = magEst_mask * params_lyot['diamPupil'] / params_mask['diamHighResMaskRef']    

    print('*** RESULTS ***')
    print(f'Bandpass = {subband_name}')
    print('Unmasked pupil clocking: %.3f deg' % clockEst_pupil)
    print('Unmasked pupil x-offset from array center pixel: %.3f pixels' % xOffset_pupil)
    print('Unmasked pupil y-offset from array center pixel: %.3f pixels' % yOffset_pupil)
    print('Magnification factor relative to the 1000x1000 reference file: %.5f' % mag_lyot_model)
    print('Magnification factor relative to the beam: %.5f' % magEst_mask)
    print('SPC Spec Lyot stop clocking relative to EXCAM grid: %.3f deg' % clockEst_mask)
    print('HLC NFOV Lyot stop clocking relative to EXCAM grid: %.3f deg' % clockEst_Lyot_HLC_NFOV)
    print('SPC WFOV Lyot stop clocking relative to EXCAM grid: %.3f deg' % clockEst_Lyot_SPC_WFOV)
    print('SPC Spec Rot Lyot stop clocking relative to EXCAM grid: %.3f deg' % clockEst_Lyot_SPC_SPEC_Rot)
    print('Clocking of HLC NFOV Lyot stop relative to pupil: %.3f deg' % (clockEst_Lyot_HLC_NFOV - clockEst_pupil))

    # Optionally write outputs to a specified file
    if write:
        outdict = {
            'clocking_pupil': float(clockEst_pupil),
            'diam_pupil': float(diamEst_pupil),
            'xOffsetPupil': float(xOffset_pupil),
            'yOffsetPupil': float(yOffset_pupil),
            'lyot_mag_vs_ref': float(mag_lyot_model),
            'lyot_mag_vs_pupil': float(magEst_mask),
            'clocking_lyot_all': float(clockEst_mask),
            'clocking_lyot_spc_spec': float(clockEst_mask),
            'clocking_lyot_hlc_nfov': float(clockEst_Lyot_HLC_NFOV),
            'clocking_lyot_spc_wfov': float(clockEst_Lyot_SPC_WFOV),
            'clocking_lyot_spc_spec_rot': float(clockEst_Lyot_SPC_SPEC_Rot),
            'lam': float(lam)
        }

        fn_out = os.path.join(MODEL_PATH, 'band%d' % band_num, 'params_meas_lsam_calib_band%d.yaml' % band_num)
        writeyaml(outdict=outdict, path=abs_to_dorba(fn_out))
        print(f'Lyot stop and pupil best-fit values printed to: {fn_out}')

    # Optionally display the reference and measured psf
    if show_plot:

        # let's read and plot the mask model as well, why not?
        mask_model = fits.getdata(os.path.join(
            DATA_PATH_SPEC, params_mask['fnMaskRefHighRes']))

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0,].imshow(pupil_img, cmap='gray')
        ax[0,].set_title('Unmasked Pupil')

        ax[1,].imshow(masked_img, cmap='gray')
        ax[1,].set_title('Masked Pupil')

        ax[2,].imshow(mask_model, cmap='gray')
        ax[2,].set_title('Mask Model')

        for hax in ax:
            hax.invert_yaxis()

        block = hasattr(main, '__file__')  # block if run from command line
        if block:
            print('close figure to continue...')

        plt.show(block=block)

    # Place outputs in a dictionary for easier reference
    out = {}
    out['pupil'] = {}
    out['pupil']['clocking'] = clockEst_pupil
    out['pupil']['xOffset'] = xOffset_pupil
    out['pupil']['yOffset'] = yOffset_pupil
    out['lyot'] = {}
    out['lyot']['clocking'] = clockEst_mask
    out['lyot']['mag'] = mag_lyot_model

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Estimate the magnification and clocking of the Lyot stops.",
                                     description="Estimate the Lyot stops' magnification and rotation from two EXCAM pupil imags.")
    parser.add_argument("frame_id_unmasked", type=int,
                        help="EDS frame ID for the unmasked pupil image.")
    parser.add_argument("frame_id_masked", type=int,
                        help="EDS frame ID for the pupil image with the SPC Spec Lyot stop in the beam.")
    parser.add_argument("subband_name", type=str, choices=BAND_CHOICES,
                        help=f"Which color filter the images were taken with. Choices are {BAND_CHOICES}")
    parser.add_argument("--no_write", action="store_true",
                        help="Default action is to write the outputs to a YAML file. Use this option not to write.")
    parser.add_argument("--fn_offset_params", type=str, default=FN_OFFSET_PARAMS_DEFAULT,
                        help="Full path within the docker container to the stored config file. Default is '" + FN_OFFSET_PARAMS_DEFAULT + "'.")
    parser.add_argument("--fn_lyot_calib", type=str, default=FN_LYOT_CALIB_DEFAULT,
                        help="Full path within the docker container to the stored config file. Default is '" + FN_LYOT_CALIB_DEFAULT + "'.")
    parser.add_argument("--data_path_pupilfit", type=str, default=DATA_PATH_PUPILFIT_DEFAULT,
                        help="Directory to serve as a base for relative paths with YAML files. Default is '" + DATA_PATH_PUPILFIT_DEFAULT + "'.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Default action is to display a graphic of the PSFs. Use this option to bypass any graphic display.")

    args = parser.parse_args()

    retvals = compute_lsam_mag_clocking(
        args.frame_id_unmasked,
        args.frame_id_masked,
        args.subband_name,
        write=(not args.no_write),
        fn_lyot_calib=args.fn_lyot_calib,
        fn_offset_params=args.fn_offset_params,
        data_path_pupilfit=args.data_path_pupilfit,
        show_plot=(not args.no_plot),
    )
