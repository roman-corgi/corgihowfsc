"""
Create sinc-sinc-sine probes that fit within the SPM and LS openings.

Plots are generated that show the DM surface and the pupil mask openings.

NOTES on data types:
    Refer to https://note.nkmk.me/en/python-numpy-dtype-astype/
    i1 is uint8
    f4 is float32 (f for float, and 4 for 4 bytes)
    < = little-endian (LSB first)
    > = big-endian (MSB first)

Example Calls in a Bash Terminal:
python write_sinc_probes.py --mode 'nfov_band1' --dark_hole '360deg'
python write_sinc_probes.py --mode 'nfov_band1_half' --dark_hole 'top'
python write_sinc_probes.py --mode 'spec_band3' --dark_hole 'both_sides' --write
python write_sinc_probes.py --mode 'wfov_band4' --dark_hole '360deg' --write

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
from howfsc.util.prop_tools import make_dmrel_probe

# PATHS
HERE = os.path.dirname(os.path.abspath(__file__))
thisFolder = os.path.basename(HERE)
howfscpath = os.path.dirname(os.path.abspath(corgihowfsc.__file__))
probepath = os.path.join(howfscpath, 'model', 'probes')

# SIT_ABS_PATH = os.path.join(HERE.split('cgi-sit')[0], 'cgi-sit')
# UTIL_DIR = os.path.join(SIT_ABS_PATH, 'utilities')
# TVAC_ABS_PATH = os.path.join(SIT_ABS_PATH, 'tvac')
# AAC_ABS_PATH = os.path.join(TVAC_ABS_PATH, 'aac')
# AAC_REL_PATH = os.path.relpath(AAC_ABS_PATH, start=GSW_ABS_PATH)
# DM1_ABS_PATH = os.path.join(TVAC_ABS_PATH, 'aac', 'dm1')
# DM2_ABS_PATH = os.path.join(TVAC_ABS_PATH, 'aac', 'dm2')
# SEED_ABS_PATH = os.path.join(TVAC_ABS_PATH, 'aac', 'seed')
# print('SEED_ABS_PATH = %s' % SEED_ABS_PATH)



# # Default Values:
# DEFAULT_CORO_MODE = 'wfov_band1'
# TVAC_REL_PATH = os.path.join('data', USER, 'cgi-sit', 'tvac')
# PARAM_REL_PATH = os.path.join('data', USER, 'cgi-sit', 'tvac', 'aac', 'any_band')

# MASK_REL_PATH = os.path.join(TVAC_REL_PATH, 'aac', 'mask_designs', 'SPC_20200610_WFOV')
# MASK_ABS_PATH = os.path.join(GSW_ABS_PATH, MASK_REL_PATH)

# TVAC_ABS_PATH = os.path.join(GSW_ABS_PATH, TVAC_REL_PATH)
# PARAM_ABS_PATH = os.path.join(GSW_ABS_PATH, PARAM_REL_PATH)

# OUT_BIN_PATH = os.path.join(TVAC_ABS_PATH, 'tv-30', 'bin')
# OUT_FITS_PATH = os.path.join(TVAC_ABS_PATH, 'tv-30', 'fits')
# OUT_PNG_PATH = os.path.join(TVAC_ABS_PATH, 'tv-30', 'python', 'output')

# FT_DIR = 'reverse'

NI_DESIRED_DEFAULT = 1e-5
MAXSEP_DEFAULT = 22.0
DELTAX_ACT_DEFAULT = None #0
DELTAY_ACT_DEFAULT = None #16
# ROT_LIST_DEFAULT = [0, 0, 90]
# PHASE_LIST_DEFAULT = [90, 0, 0]

# %% Functions
def fft2(arrayIn):
    """Perform an energy-conserving 2-D FFT including fftshift."""
    arrayOut = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayIn)))/np.sqrt(arrayIn.shape[0]*arrayIn.shape[1])

    return arrayOut


def write_sinc_probes(
        mode, dark_hole, ni_desired=NI_DESIRED_DEFAULT, #maxsep=MAXSEP_DEFAULT,
        deltax_act=DELTAX_ACT_DEFAULT, deltay_act=DELTAY_ACT_DEFAULT,
        write=False,
        #rot_list=ROT_LIST_DEFAULT, phase_list = PHASE_LIST_DEFAULT,
        ):
    
    # # Convert the string-ified list of frame ID numbers into an actual list of integers
    # # Blame this on https://www.freecodecamp.org/news/python-string-to-array-how-to-convert-text-to-a-list/
    # if isinstance(rot_list, str):
    #     rot_list = list(map(int, rot_list.replace(" ", "").replace("[", "").replace("]", "").split(sep=',')))
    # if isinstance(phase_list, str):
    #     phase_list = list(map(int, phase_list.replace(" ", "").replace("[", "").replace("]", "").split(sep=',')))


    if 'nfov' in mode:  # nfov_band1, nfov_band2, nfov_band3, nfov_band4
        if '360' in dark_hole:  # 360-degree dark zone
            rot_list = [0, 0, 90]
            phase_list = 90 + np.array([0, 60, 120]) # [90, 0, 0] 
            deltax_act = 0 #7 #0
            deltay_act = 16 #13 #16
        else:  # half dark zone
            rot_list = [0, 0, 0] # [90, 90, 90]#
            phase_list = 90 + np.array([0, 60, 120])
            deltax_act = 0
            deltay_act = 16

        Rin = 2.8
        Rout = 209.7
        ximin = 0
        ximax = Rout+1
        etamax = ximax
        etamin = -etamax
        lod_min = Rin
        lod_max = Rout

    elif 'wfov' in mode:  # wfov_band1, wfov_band4
        rot_list = [0, 0, 90]
        phase_list = 90 + np.array([0, 60, 120])
        deltax_act = 0
        deltay_act = 16

        Rin = 5.6
        Rout = 20.4
        ximin = 0
        ximax = Rout+1
        etamax = ximax
        etamin = -etamax
        lod_min = Rin
        lod_max = Rout

    elif 'spec' in mode:  # SPEC and SPECROT
        if 'rot' in mode:  # SPECROT only
            specrot_angle = 60  # TODO: Verify that it is 60 and not 120
            rot_list = 3*[specrot_angle, ]  
            phase_list = 90 + np.array([0, 60, 120])
            if deltax_act is None or deltay_act is None:
                deltax_act = 16*np.sin(np.deg2rad(specrot_angle))
                deltay_act = 16*np.cos(np.deg2rad(specrot_angle))
        else:  # SPEC only
            rot_list = [0, 0, 0]
            phase_list = 90 + np.array([0, 60, 120])
            deltax_act = 0
            deltay_act = 16

        Rin = 2.6
        Rout = 9.4
        ximin = Rin-1
        ximax = Rout+1
        etamax = Rout*np.sin(np.deg2rad(65/2)) + 1
        etamin = -etamax
        lod_min = Rin
        lod_max = Rout


    args = get_args(
        mode=mode,
        dark_hole=dark_hole,
    )
    modelpath, cfgfile, jacfile, cstratfile, probefiles, hconffile, n2clistfiles, dmstartmaps = load_files(args, howfscpath)
    cfg = CoronagraphMode(cfgfile)
    dmlist = cfg.initmaps
    homf_dict = loadyaml(cfgfile)
    dmreg_dm1 = homf_dict['dms']['DM1']['registration']
    
    # # From FALCO:
    # surfMax = 4*pi*mp.lambda0*sqrt(ni_desired); % [meters]
    # probeHeight = surfMax * sinc(width*XS) .* sinc(height*YS) .* cos(2*pi*(xiOffset*XS + etaOffset*YS) + phaseShift);
    lam = homf_dict['sls'][1]['lam']
    surfMax = 4*np.pi*lam*np.sqrt(ni_desired)  # [meters]
    print('Max probe height in meters to get NI=%.2e is: %.4g' % (ni_desired, surfMax))

    diam_pupil_pix = homf_dict['sls'][1]['epup']['pdp']
    dact = diam_pupil_pix/dmreg_dm1['ppact_cx']

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
    usable_act_map[tiemap==0] = True

    # # # Nonlinear surface calculation using compute_delta_height_map_from_command_maps()
    # # fnRefCubeDM1_tmp = os.path.join(DM1_ABS_PATH, 'height_cube_integrated.fits')
    # # fnRefCommandVecDM1_tmp = os.path.join(DM1_ABS_PATH, 'height_cube_command_vec.fits')
    # # fnRefCubeDM2_tmp = os.path.join(DM2_ABS_PATH, 'height_cube_integrated.fits')
    # # fnRefCommandVecDM2_tmp = os.path.join(DM2_ABS_PATH, 'height_cube_command_vec.fits')

    # dm1_dh_m = compute_delta_height_map_from_command_maps(
    #     dm1v_init, dm1v_end, fnRefCubeDM1_tmp,
    #     fnRefCommandVecDM1_tmp, crosstalk_fn=crosstalk_fn_dm1)
    # dm2_dh_m = compute_delta_height_map_from_command_maps(
    #     dm2v_init, dm2v_end, fnRefCubeDM2_tmp,
    #     fnRefCommandVecDM2_tmp, crosstalk_fn=crosstalk_fn_dm2)

    # inf_func_dm1 = fits.getdata(os.path.join(modelpath, dmreg_dm1['inffn']))

    # fn_gain_map_dm1 = os.path.join(modelpath, homf_dict['dms']['DM1']['voltages']['gainfn'])

    # gain_map_dm1 = fits.getdata(fn_gain_map_dm1)

    # poke_grid = np.zeros((NACT, NACT))
    # poke_grid[9, 23] = 20  # 3 o'clock
    # poke_grid[39, 24] = 20 # 9 o'clock
    # poke_grid[16, 10] = 20 # 1  o'clock
    # poke_grid[16, 37] = 20 # 5  o'clock
    # poke_grid[31, 10] = 20 # 11  o'clock
    # poke_grid[31, 37] = 20 # 7  o'clock
    
    # # Save poke grid to FITS and .bin files
    # fn_dm1_poke_grid = os.path.join(DM1_ABS_PATH, 'dmrel_wfov_poke_grid.fits')
    # fn_dm2_poke_grid = os.path.join(DM2_ABS_PATH, 'dmrel_wfov_poke_grid.fits')
    # fits.writeto(fn_dm1_poke_grid, poke_grid, overwrite=True)
    # fits.writeto(fn_dm2_poke_grid, poke_grid, overwrite=True)
    # fits2bin('float32', fn_dm1_poke_grid)
    # fits2bin('float32', fn_dm2_poke_grid)
    
    # probe_center_angle = -60
    # sep = 15 # 15 to 16
    # print(np.array([[np.cos(np.radians(probe_center_angle)), np.sin(np.radians(probe_center_angle))], [-np.sin(np.radians(probe_center_angle)), np.cos(np.radians(probe_center_angle))]]) @ np.array([0, sep]))
        
    # xcenter = -13
    # ycenter = 7


    # xcenter_list = [0, 0, 13, 14, -14, -13]
    # ycenter_list = [16, -15, 7, -8, -8, 7]
    # oclock_list = [12, 6, 2, 4, 8, 10]
    # opening_size = ['big', 'small', 'small', 'big', 'big', 'small']  # for reference only

    # probe_name_list = ['cos', 'sinlr', 'sinud']
    nprobepairs = 3
    probe_name_list = []
    for ii in range(nprobepairs):
        rot = rot_list[ii]
        sin_phase = phase_list[ii]
        probe_name = 'sin%d_rot%d' % (sin_phase, rot)
        probe_name_list.append(probe_name)
    
    probe_ni_map_list = []

    for index_phase, sin_phase in enumerate(phase_list):
        print('*** Probe %d ***' % index_phase)
        
        rot = rot_list[index_phase]
        probe_name = probe_name_list[index_phase]
        
        probe_tuple = make_dmrel_probe(
            cfg=cfg, dmlist=dmlist, dact=dact, xcenter=deltax_act, ycenter=deltay_act, clock=rot,
            ximin=ximin, ximax=ximax,
            etamin=etamin, etamax=etamax,
            phase=sin_phase, target=ni_desired, lod_min=lod_min, lod_max=lod_max,
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

        plt.figure(2+10*index_phase)
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
        overlay1 = 3*dm_surf/np.max(dm_surf) + pupil_masks/np.max(pupil_masks)
    
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

        ft_dm_surf = inin(fft2(inin(lyot*dm_surf, (1024, 1024))), (200, 200))

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


#####################################################################
if __name__ == '__main__':
    # supported_coro_mode_list = ('wfov_band1', 'sim_wfov_band1')

    parser = argparse.ArgumentParser(prog="python write_sinc_probes.py",
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
    # parser.add_argument("--rot_list", type=str, default=ROT_LIST_DEFAULT,
    #                     help=f"List of the rotations of the three probes. Units of degrees. Default is {ROT_LIST_DEFAULT}.")
    # parser.add_argument("--phase_list", type=str, default=PHASE_LIST_DEFAULT,
    #                     help=f"List of the sine phases of the three probes. Units of degrees. Default is {PHASE_LIST_DEFAULT}.")

    args = parser.parse_args()
    
    mode = args.mode
    dark_hole = args.dark_hole

    write_sinc_probes(
        mode, dark_hole, ni_desired=args.ni, # maxsep=args.maxsep,
        deltax_act=args.deltax, deltay_act=args.deltay, write=args.write,
        # rot_list=args.rot_list, phase_list=args.phase_list,
    )
