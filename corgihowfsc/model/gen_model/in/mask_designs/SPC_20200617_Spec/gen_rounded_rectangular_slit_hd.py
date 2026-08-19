"""
Make 2-D, gray-edged, rounded-end slit transmission maps in 2-D arrays.

Steps:
1. Create a binary version of the mask
2. Find all the edge pixels
3. Upsample at all the edge pixels to compute the fractional value at each.
4. Save to the filename, if given.

"""
import math
import os

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# from gsw.cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample

# # dorba and import cal tools
# import ctc.dorba
# ctc.dorba.init(data_subdir=os.environ['USER'])

HERE = os.path.dirname(os.path.abspath(__file__))

# %% Optical system values -- Do not change

aoiDeg = 5.0  # [degrees]
aoiAxis = 'x'  # 'x' or 'y'
fnum = 41.626
lam1um = 0.575
lam2um = 0.660
lam3um = 0.730
lam4um = 0.825
flamD1 = fnum*lam1um  # [microns]
flamD2 = fnum*lam2um  # [microns]
flamD3 = fnum*lam3um  # [microns]
flamD4 = fnum*lam4um  # [microns]
flamD2pt5 = fnum*(lam2um+lam3um)/2.0

ppl1 = 2.262
ppl2pt5 = ppl1*(695/575)  # pixels per lambda/D
ppl3 = ppl1*(730/575)  # pixels per lambda/D

# %% Mask Values

# Slits to model: R3C1 (first null) and R1C2 (FWHM)

# R3C1 (first null) 
# % SPC ~1st Null Vertical Slits @ average of Bands 2 and 3 (695nm):
# % Positions 11-12
w_first_null_slit = 4*ppl2pt5  # pixels
h_first_null_slit = 12*ppl2pt5  # pixels
roc_first_null_slit = (347.16/12.0/30.0)*ppl2pt5

# R1C2 (FWHM)
# Band 3 Vertical Slits: Positions 2, 21


# w_fwhm_slit = 2.0*ppl3  # pixels
# h_fwhm_slit = 12*ppl3  # pixels
# roc_fwhm_slit = 1.0*ppl3  # pixels


# %%

def gen_rounded_rectangular_slit(width, height, roc, upsampleFactor=101, padFac=1.2,
                                 rotDeg=0, xOffset=0, yOffset=0, magInternal=10,
                                 fn_out=None):
    # width and height are in pixels

    width *= magInternal
    height *= magInternal
    roc *= magInternal

    Narray = (2 * math.ceil(padFac/2*(np.max([width, height])))) + 1  # Make odd

    # Coordinates (at low resolution)
    xs = np.linspace(-(Narray-1)/2, (Narray-1)/2, Narray)
    ys = xs
    [XS, YS] = np.meshgrid(xs, ys)
    RS2 = XS**2 + YS**2
    RS = np.sqrt(RS2)

    width_short = width
    height_short = height - 2*roc

    width_narrow = width - 2*roc
    height_narrow = height

    xc_circle_ul = -(width/2 - roc)
    yc_circle_ul = height/2 - roc

    xc_circle_ur = width/2 - roc
    yc_circle_ur = height/2 - roc

    xc_circle_br = width/2 - roc
    yc_circle_br = -(height/2 - roc)

    xc_circle_bl = -(width/2 - roc)
    yc_circle_bl = -(height/2 - roc)

    maskInnerShort = np.zeros_like(XS)
    maskInnerShort = (
        (XS >= -width_short/2) &
        (XS <= width_short/2) &
        (YS >= -height_short/2) &  
        (YS <= height_short/2)
        )

    maskInnerNarrow = np.zeros_like(XS)
    maskInnerNarrow = (
        (XS >= -width_narrow/2) &
        (XS <= width_narrow/2) &
        (YS >= -height_narrow/2) &  
        (YS <= height_narrow/2)
        )

    maskSharpBinNegative = np.ones_like(RS)

    maskSharpBinNegative = (
        ((XS-xc_circle_ul)**2 + (YS-yc_circle_ul)**2 > roc**2) &
        ((XS-xc_circle_ur)**2 + (YS-yc_circle_ur)**2 > roc**2) &
        ((XS-xc_circle_br)**2 + (YS-yc_circle_br)**2 > roc**2) &
        ((XS-xc_circle_bl)**2 + (YS-yc_circle_bl)**2 > roc**2) &
        ~(maskInnerShort | maskInnerNarrow)
        )

    maskSharpBin = ~maskSharpBinNegative

#     plt.figure(11)
#     plt.imshow(maskSharpBin)
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.pause(1e-2)
#     plt.show()

    maskSharp = maskSharpBin.astype(float)
    maskRoundedBin = maskSharpBin
    maskRounded = maskRoundedBin.astype(float)

    kernel = np.ones((3, 3))

    roundedTemp = convolve2d(maskRounded, kernel, mode='same') / np.sum(kernel)
    grayIndsRounded = np.nonzero(np.logical_and(roundedTemp > 0, roundedTemp < 1))
    roundedEdges = np.zeros_like(RS)
    roundedEdges[grayIndsRounded] = True

    # annulusTemp = convolve2d(maskSharp, kernel, mode='same') / np.sum(kernel)
    # grayIndsAnnulus = np.nonzero(np.logical_and(annulusTemp > 0, annulusTemp < 1))
    # annularEdges = np.zeros_like(RS)
    # annularEdges[grayIndsAnnulus] = True


    # %% Upsampling

    dx = 1  # /res
    dxUp = dx/upsampleFactor
    xUp = np.linspace(-(upsampleFactor-1)/2, (upsampleFactor-1)/2, upsampleFactor)*dxUp
    [Xup0, Yup0] = np.meshgrid(xUp, xUp)

    subpixel = np.zeros((upsampleFactor, upsampleFactor))
    # allFillets = np.zeros((Narray, Narray))
    allShapes = 1 - maskRounded.copy()
    # annulusGray = maskSharp.copy()

    # Make the grayscale versions of the fillets
    for ii in range(len(grayIndsRounded[0])):

        row = grayIndsRounded[0][ii]
        col = grayIndsRounded[1][ii]

        subpixel = 0*subpixel

        Xup = Xup0 + XS[row, col]
        Yup = Yup0 + YS[row, col]
        RHOSupUL = np.sqrt((Xup-xc_circle_ul)**2 + (Yup-yc_circle_ul)**2)
        RHOSupUR = np.sqrt((Xup-xc_circle_ur)**2 + (Yup-yc_circle_ur)**2)
        RHOSupBR = np.sqrt((Xup-xc_circle_br)**2 + (Yup-yc_circle_br)**2)
        RHOSupBL = np.sqrt((Xup-xc_circle_bl)**2 + (Yup-yc_circle_bl)**2)

        maskInnerShortUp = (
            (Xup >= -width_short/2) &
            (Xup <= width_short/2) &
            (Yup >= -height_short/2) &  
            (Yup <= height_short/2)
        )

        maskInnerNarrowUp = (
            (Xup >= -width_narrow/2) &
            (Xup <= width_narrow/2) &
            (Yup >= -height_narrow/2) &  
            (Yup <= height_narrow/2)
        )

        subpixelNegative = (
            (RHOSupUL > roc**2) &
            (RHOSupUR > roc**2) &
            (RHOSupBR > roc**2) &
            (RHOSupBL > roc**2) &
            ~(maskInnerShortUp | maskInnerNarrowUp)
        )

        subpixel = ~subpixelNegative

        pixelValue = np.sum(subpixel.astype(int)) / upsampleFactor**2
        allShapes[row, col] = pixelValue

    maskRoundedGray = 1-allShapes

    maskOut = downsample(
        maskIn=maskRoundedGray,
        rotDeg=rotDeg,
        mag=1/magInternal,
        xOffset=xOffset,
        yOffset=yOffset,
        padFac=padFac,
        flipx=False,
        )

    # Get rid of floating point noise for zero values
    thresh = 10 * np.finfo(float).eps
    maskOut[np.abs(maskOut) < thresh] = 0

#     # %% Plotting and output writing
#
#     plt.figure(12)
#     plt.imshow(maskRoundedGray)
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.pause(1e-2)
#
#     plt.figure(11)
#     plt.imshow(maskOut)
#     plt.gca().invert_yaxis()
#     plt.colorbar()
#     plt.pause(1e-2)
#
#     plt.show()

    if fn_out is not None:
        fits.writeto(fn_out, maskOut, overwrite=True)

    return maskOut


if __name__ == '__main__':


    # # R1C2 (FWHM)
    # # Band 3 Vertical Slits: Positions 2, 21
    # fs_pos = 'r1c2'
    # ppl = 100 #ppl3  # EXCAM resolution
    # w_fwhm_slit = 2.0*ppl  # pixels
    # h_fwhm_slit = 12*ppl  # pixels
    # roc_fwhm_slit = 1.0*ppl  # pixels

    # gen_rounded_rectangular_slit(
    #     w_fwhm_slit,
    #     h_fwhm_slit,
    #     roc_fwhm_slit,
    #     magInternal=10,
    #     rotDeg=90,
    #     fn_out=os.path.join(HERE, 'fsam_%s_ppl%.2f.fits' % (fs_pos, ppl)),
    # )

#    # R4C6 (FWHM)
#    # Band 3 Vertical Slits: Positions 2, 21
#    fs_pos = 'r4c6'
#    ppl = 100
#    w_fwhm_slit = 2.0*ppl  # pixels
#    h_fwhm_slit = 6*ppl  # pixels
#    roc_fwhm_slit = 1.0*ppl  # pixels
#    gen_rounded_rectangular_slit(
#        w_fwhm_slit,
#        h_fwhm_slit,
#        roc_fwhm_slit,
#        magInternal=10,
#        rotDeg=90,
#        fn_out=os.path.join(HERE, 'fsam_%s_ppl%.2f.fits' % (fs_pos, ppl)),
#    )
    
#    # R2C1
#    # MSWC field stop, 9x9 lambda/D, Band 1
#    fs_pos = 'r2c1'
#    ppl = 100
#    w_fwhm_slit = 9.0*ppl  # pixels
#    h_fwhm_slit = 9.0*ppl  # pixels
#    roc_fwhm_slit = (215.4/9/30)*ppl  # pixels
#    gen_rounded_rectangular_slit(
#        w_fwhm_slit,
#        h_fwhm_slit,
#        roc_fwhm_slit,
#        magInternal=10,
#        rotDeg=0,
#        fn_out=os.path.join(HERE, 'fsam_%s_ppl%.2f.fits' % (fs_pos, ppl)),
#    )
    
    # R1C4
    # MSWC field stop, 9x9 lambda/D, Band 1
    fs_pos = 'r1c4'
    ppl = 100
    w_fwhm_slit = 9.0*ppl  # pixels
    h_fwhm_slit = 9.0*ppl  # pixels
    roc_fwhm_slit = (309.1/9/30)*ppl  # pixels
    gen_rounded_rectangular_slit(
        w_fwhm_slit,
        h_fwhm_slit,
        roc_fwhm_slit,
        magInternal=10,
        rotDeg=0,
        fn_out=os.path.join(HERE, 'fsam_%s_ppl%.2f.fits' % (fs_pos, ppl)),
    )

    # # Make sure to match EXCAM orientation. In this case can just rotate 90 degrees.
    # gen_rounded_rectangular_slit(
    #     w_first_null_slit,
    #     h_first_null_slit,
    #     roc_first_null_slit,
    #     magInternal=10,
    #     rotDeg=90,
    #     fn_out=os.path.join(HERE, 'fsam_r3c1_excam_res.fits'),
    # )
