"""
Create Gaussian probes that correspond to the original sinc-sinc-sine probes,
which have been adjusted to fit within the SPM and LS openings.

Plots are generated that show the DM surface and the pupil mask openings.

NOTES on data types:
    Refer to https://note.nkmk.me/en/python-numpy-dtype-astype/
    i1 is uint8
    f4 is float32 (f for float, and 4 for 4 bytes)
    < = little-endian (LSB first)
    > = big-endian (MSB first)

Example Calls in a Bash Terminal:
python write_gaussian_probes.py --mode 'nfov_band1' --dark_hole '360deg'
python write_gaussian_probes.py --mode 'nfov_band1' --dark_hole 'half_top'
python write_gaussian_probes.py --mode 'spec_band3' --dark_hole 'both_sides' --write
python write_gaussian_probes.py --mode 'wfov_band4' --dark_hole '360deg' --write

"""
import argparse
import os
import sys

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

import corgihowfsc
from corgihowfsc.utils.howfsc_initialization import get_args, load_files

from howfsc.model.mode import CoronagraphMode
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.insertinto import insertinto as inin
from howfsc.util.loadyaml import loadyaml
from corgihowfsc.utils.cgi_prop_tools import make_dmrel_probe_gaussian

# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
thisFolder = os.path.basename(HERE)
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
probepath = os.path.join(howfscpath, 'model', 'probes')

NI_DESIRED_DEFAULT = 1e-5
MAXSEP_DEFAULT = 22.0
DELTAX_ACT_DEFAULT = None  # 0
DELTAY_ACT_DEFAULT = None  # 16


# %% Functions
def fft2(arrayIn):
    """Perform an energy-conserving 2-D FFT including fftshift."""
    arrayOut = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayIn))) / np.sqrt(arrayIn.shape[0] * arrayIn.shape[1])

    return arrayOut


def write_gaussian_probes(
        mode, dark_hole, ni_desired=NI_DESIRED_DEFAULT,  # maxsep=MAXSEP_DEFAULT,
        deltax_act=DELTAX_ACT_DEFAULT, deltay_act=DELTAY_ACT_DEFAULT,
        write=False,
        # rot_list=ROT_LIST_DEFAULT, phase_list = PHASE_LIST_DEFAULT,
):
    # # Convert the string-ified list of frame ID numbers into an actual list of integers
    # # Blame this on https://www.freecodecamp.org/news/python-string-to-array-how-to-convert-text-to-a-list/
    # if isinstance(rot_list, str):
    #     rot_list = list(map(int, rot_list.replace(" ", "").replace("[", "").replace("]", "").split(sep=',')))
    # if isinstance(phase_list, str):
    #     phase_list = list(map(int, phase_list.replace(" ", "").replace("[", "").replace("]", "").split(sep=',')))

    if 'nfov' in mode:  # nfov_band1, nfov_band2, nfov_band3, nfov_band4
        if '360' in dark_hole:  # 360-degree dark zone
            sigma = 1.5
            deltax_act_list = [13, 13, 14]
            deltay_act_list = [8, 9, 9]
        else:  # half dark zone
            sigma = 1.5
            deltax_act = 0
            deltay_act = 16

        Rin = 2.8
        Rout = 209.7
        lod_min = Rin
        lod_max = Rout

    elif 'wfov' in mode:  # wfov_band1, wfov_band4
        sigma = 1.5
        deltax_act_list = [13, 13, 14]
        deltay_act_list = [8, 9, 9]

        Rin = 5.6
        Rout = 20.4
        lod_min = Rin
        lod_max = Rout

    """ unsupported mode for Gaussian probes for now, but here is a starting point for the parameters as used for sinc probes:
    elif 'spec' in mode:  # SPEC and SPECROT
        if 'rot' in mode:  # SPECROT only
            specrot_angle = 60  # TODO: Verify that it is 60 and not 120
            sigma = 1.5
            if deltax_act is None or deltay_act is None:
                deltax_act = 16 * np.sin(np.deg2rad(specrot_angle))
                deltay_act = 16 * np.cos(np.deg2rad(specrot_angle))
        else:  # SPEC only
            sigma = 1.5
            deltax_act = 0
            deltay_act = 16

        Rin = 2.6
        Rout = 9.4
        ximin = Rin - 1
        ximax = Rout + 1
        etamax = Rout * np.sin(np.deg2rad(65 / 2)) + 1
        etamin = -etamax
        lod_min = Rin
        lod_max = Rout
    """

    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, _probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args,
                                                                                                           howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps
    homf_dict = loadyaml(cfgfile)
    dmreg_dm1 = homf_dict['dms']['DM1']['registration']

    # # From FALCO:
    lam = homf_dict['sls'][1]['lam']
    surfMax = 4 * np.pi * lam * np.sqrt(ni_desired)  # [meters]
    print('Max probe height in meters to get NI=%.2e is: %.4g' % (ni_desired, surfMax))

    diam_pupil_pix = homf_dict['sls'][1]['epup']['pdp']
    dact = diam_pupil_pix / dmreg_dm1['ppact_cx']

    fn_amp = os.path.join(modelpath, homf_dict['sls'][1]['epup']['afn'])
    fn_ph = os.path.join(modelpath, homf_dict['sls'][1]['epup']['pfn'])
    fn_lyot = os.path.join(modelpath, homf_dict['sls'][1]['lyot']['afn'])
    fn_spm = os.path.join(modelpath, homf_dict['sls'][1]['sp']['afn'])
    fn_fpm = os.path.join(modelpath, homf_dict['sls'][1]['fpm']['afn'])
    fn_dh = os.path.join(modelpath, homf_dict['sls'][1]['dh'])

    amp = fits.getdata(fn_amp)
    ph = fits.getdata(fn_ph)
    lyot = fits.getdata(fn_lyot)
    spm = fits.getdata(fn_spm)
    fpm = fits.getdata(fn_fpm)
    dh = fits.getdata(fn_dh)

    # NXFRESNEL = spm.shape[0] # 2048

    # dm_temp = loadyaml(path=dorbaify_path(FN_HOWFSC_YAML))
    # dmreg_dm1 = dm_temp['dms']['DM1']['registration']
    # dmreg_dm2 = dm_temp['dms']['DM2']['registration']

    # # Read in tie map
    # which_dm = 1
    NACT = homf_dict['dms']['DM1']['registration']['nact']
    # VMIN_DEFAULT = homf_dict['dms']['DM1']['voltages']['vmin']
    # VMAX_DEFAULT = homf_dict['dms']['DM1']['voltages']['vmax']
    tiemap = fits.getdata(os.path.join(modelpath, homf_dict['dms']['DM1']['voltages']['tiefn']))
    usable_act_map = np.zeros((NACT, NACT), dtype=bool)
    usable_act_map[tiemap == 0] = True

    probe_name_list = ['gauss1', 'gauss2', 'gauss3']
    probe_ni_map_list = []

    for index_probe, _ in enumerate(deltax_act_list):
        print('*** Probe %d ***' % index_probe)

        deltax_act = deltax_act_list[index_probe]
        deltay_act = deltay_act_list[index_probe]
        probe_name = probe_name_list[index_probe]

        probe_tuple = make_dmrel_probe_gaussian(
            cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act, ycenter=deltay_act, sigma=sigma,
            target=ni_desired, lod_min=lod_min, lod_max=lod_max,
            ind=1, maxiter=5)

        dpv = probe_tuple[0]
        probe_ni_map = probe_tuple[1]
        dh_mask = probe_tuple[2]
        dm_surf = probe_tuple[3]
        pupil_masks = probe_tuple[4]

        probe_ni_map_list.append(probe_ni_map)

        dpv = usable_act_map * dpv

        plt.figure(1)
        plt.clf()
        plt.imshow(dpv)
        plt.title('Probe DMREL Map')
        plt.gca().invert_yaxis()
        plt.colorbar()

        plt.figure(3)
        plt.clf()
        plt.title('Scoring Region')
        plt.imshow(dh_mask.astype(int))
        plt.gca().invert_yaxis()
        plt.colorbar()

        plt.figure(2 + 10 * index_probe)
        # plt.clf()
        plt.imshow(probe_ni_map)
        plt.title('Probe-only Intensity')
        plt.gca().invert_yaxis()
        plt.colorbar()

        # plt.figure(4)
        # plt.clf()
        # plt.imshow(dm_surf)
        # plt.gca().invert_yaxis()
        # plt.colorbar()

        # plt.figure(5)
        # plt.clf()
        # plt.imshow(pupil_masks)
        # plt.gca().invert_yaxis()
        # plt.colorbar()

        if write:
            fn_probe_base = 'dmrel_%s_%s_ni%.0e_sin%d_rot%d' % (mode, dark_hole, ni_desired, sin_phase, rot)
            fn_probe_fits = os.path.join(probepath, fn_probe_base + '.fits')
            fn_probe_png = os.path.join(probepath, fn_probe_base + '.png')
            fn_probe_bin = os.path.join(probepath, fn_probe_base + '.bin')
            fits.writeto(fn_probe_fits, dpv, overwrite=True)
            # fits2bin('float32', fn_probe_fits, fn_out=fn_probe_bin)

        # dm1_dh_m = dpv * gain_map_dm1 * usable_act_map

        # # input is actuator poke in radians, output is phase surface in radians
        # # is dmhtoph linear, i.e. dphtoph(h1) - dmhotph(h2) = dmhotph(h1 - h2) ???
        # # Note: dorba requires all arguments to be keyword (no positional arguments)
        # nrow = amp.shape[0]
        # ncol = amp.shape[1]
        # dm1_surf_m = dmhtoph(
        #     nrow=nrow, ncol=ncol, dmin=dm1_dh_m, nact=dmreg_dm1['nact'],
        #     inf_func=inf_func_dm1, ppact_d=dmreg_dm1['ppact_d'],
        #     ppact_cx=dmreg_dm1['ppact_cx'], ppact_cy=dmreg_dm1['ppact_cy'],
        #     dx=dmreg_dm1['dx'], dy=dmreg_dm1['dy'], thact=dmreg_dm1['thact'],
        #     flipx=dmreg_dm1['flipx'],
        # )

        # masks = amp*spm*lyot
        overlay1 = 3 * dm_surf / np.max(dm_surf) + pupil_masks / np.max(pupil_masks)

        # plt.figure(1)
        # plt.title('dpv')
        # plt.imshow(dpv)
        # plt.gca().invert_yaxis()
        # plt.colorbar()

        plt.figure(6)
        plt.clf()
        plt.title('Probe with Mask Overlay')
        plt.imshow(overlay1)
        plt.gca().invert_yaxis()
        if write:
            print('Saving graphic to: ', fn_probe_png)
            plt.savefig(fn_probe_png, bbox_inches='tight', pad_inches=0.1)

        ft_dm_surf = inin(fft2(inin(lyot * dm_surf, (1024, 1024))), (200, 200))

        plt.figure(7)
        plt.clf()
        plt.title('ft_dm_surf')
        plt.imshow(np.abs(ft_dm_surf))
        plt.gca().invert_yaxis()

        plt.pause(3)

        print('Close figures to continue...')
        # plt.show()

        del probe_tuple

    # PPL_FPM_CENTRAL = homf_dict['sls'][1]['fpm']['ppl']
    # LAM_CENTRAL = lam
    # PPL_MEAS_REF = 2.26

    # NPAD_FPM = int(np.ceil(2*PPL_FPM_CENTRAL * diam_pupil_pix)//2)
    # NPAD_FS = int(np.ceil(2*PPL_MEAS_REF * diam_pupil_pix)//2)
    # NOUT = 100

    # # Without FPM, without probe
    # epupout = spm * amp * np.exp(1j*ph) #* np.exp(1j*4*np.pi/LAM_CENTRAL*dm1_dh_m)
    # epupout = inin(arr0=epupout, outsize=(NPAD_FPM, NPAD_FPM))
    # efoc = fft2(epupout)
    # ifoc =  inin(arr0=np.abs(efoc), outsize=(210, 210))
    # efoc = inin(arr0=efoc, outsize=(NPAD_FPM, NPAD_FPM)) #* inin(arr0=fpm, outsize=(NPAD_FPM, NPAD_FPM))
    # elyot = inin(arr0=fft2(efoc), outsize=(NPAD_FS, NPAD_FS)) * inin(arr0=lyot, outsize=(NPAD_FS, NPAD_FS))
    # efs = inin(arr0=fft2(elyot), outsize=(NOUT, NOUT))
    # ifs = np.abs(efs)**2
    # i00 = np.max(ifs)

    # # Without probe
    # epupout = spm * amp * np.exp(1j*ph) #* np.exp(1j*4*np.pi/LAM_CENTRAL*dm1_dh_m)
    # epupout = inin(arr0=epupout, outsize=(NPAD_FPM, NPAD_FPM))
    # efoc = fft2(epupout)
    # ifoc =  inin(arr0=np.abs(efoc), outsize=(210, 210))
    # efoc = inin(arr0=efoc, outsize=(NPAD_FPM, NPAD_FPM)) * inin(arr0=fpm, outsize=(NPAD_FPM, NPAD_FPM))
    # elyot = inin(arr0=fft2(efoc), outsize=(NPAD_FS, NPAD_FS)) * inin(arr0=lyot, outsize=(NPAD_FS, NPAD_FS))
    # efs0 = inin(arr0=fft2(elyot), outsize=(NOUT, NOUT))

    # # With probe
    # dm1_surf_m = dm_surf/(2*np.pi/LAM_CENTRAL)
    # # epupout = spm * np.exp(1j*4*np.pi/LAM_CENTRAL*dm_surf) * amp * np.exp(1j*ph)
    # epupout = spm * np.exp(1j*4*np.pi/LAM_CENTRAL*dm1_surf_m) * amp * np.exp(1j*ph)
    # epupout = inin(arr0=epupout, outsize=(NPAD_FPM, NPAD_FPM))
    # efoc = fft2(epupout)
    # ifoc =  inin(arr0=np.abs(efoc)**2, outsize=(210, 210))
    # efoc = inin(arr0=efoc, outsize=(NPAD_FPM, NPAD_FPM)) * inin(arr0=fpm, outsize=(NPAD_FPM, NPAD_FPM))
    # elyot = inin(arr0=fft2(efoc), outsize=(NPAD_FS, NPAD_FS)) * inin(arr0=lyot, outsize=(NPAD_FS, NPAD_FS))
    # efs = inin(arr0=fft2(elyot), outsize=(NOUT, NOUT))

    # dh = inin(arr0=dh, outsize=(NOUT, NOUT))
    # difs = np.abs(efs - efs0)**2 / i00
    # ifs = np.abs(efs)**2 / i00

    # print('Mean probe NI = %.2g' % np.mean(difs[dh==1]))

    # plt.figure(10)
    # plt.title('dm1_surf_m')
    # plt.imshow(inin(arr0=dm1_surf_m, outsize=(amp.shape)))
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(17)
    # plt.title('difs')
    # plt.imshow(np.log10(difs*dh))
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(21)
    # plt.clf()
    # plt.imshow(probe_ni_map_list[1]-probe_ni_map_list[0])
    # plt.title('Probe-only Intensity Diff.')
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(22)
    # plt.clf()
    # plt.imshow(probe_ni_map_list[2]-probe_ni_map_list[1])
    # plt.title('Probe-only Intensity Diff.')
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="python write_gaussian_probes.py",
                                     description="Compute the HOWFSC optical model YAML and FITS files for either instrument use or deriving a model-derived DM seed.")
    parser.add_argument("--mode", required=True, type=str,
                        help=f"Coronagraph mode to make probes for.")
    parser.add_argument("--dark_hole", required=True, type=str,
                        help=f"Dark hole specification in the folder name of the mode used.")
    parser.add_argument("--ni", type=float, default=NI_DESIRED_DEFAULT,
                        help=f"Desired mean normalized intensity of the probe. Default is {NI_DESIRED_DEFAULT}.")
    # parser.add_argument("--maxsep", type=float, default=MAXSEP_DEFAULT,
    #                     help=f"Max separation (in both x and y) illuminated by the probe. Must be larger than the OWA, but being overly large hurts coverage within the dark zone. Default is {MAXSEP_DEFAULT}.")
    parser.add_argument("--deltax", type=float, default=DELTAX_ACT_DEFAULT,
                        help=f"x-offset of the probe from the center of the DM. Units of actuator widths. Default is {DELTAX_ACT_DEFAULT}.")
    parser.add_argument("--deltay", type=float, default=DELTAY_ACT_DEFAULT,
                        help=f"y-offset of the probe from the center of the DM. Units of actuator widths. Default is {DELTAY_ACT_DEFAULT}.")
    parser.add_argument("--write", action="store_true",
                        help="Write the probes out to FITS files.")

    args = parser.parse_args()

    mode = args.mode
    dark_hole = args.dark_hole

    write_gaussian_probes(
        mode, dark_hole, ni_desired=args.ni,  # maxsep=args.maxsep,
        deltax_act=args.deltax, deltay_act=args.deltay, write=args.write
    )
