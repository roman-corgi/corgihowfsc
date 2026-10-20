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

from gsw.cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask as downsample

# dorba and import cal tools
import ctc.dorba
ctc.dorba.init(data_subdir=os.environ['USER'])

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
roc_first_null_slit = w_first_null_slit/2

# R1C2 (FWHM)
# Band 3 Vertical Slits: Positions 2, 21
w_fwhm_slit = 2.0*ppl3  # pixels
h_fwhm_slit = 12*ppl3  # pixels
roc_fwhm_slit = w_fwhm_slit/2


# %%

def gen_fully_rounded_slit(width, height, upsampleFactor=101, padFac=1.2,
                           rotDeg=0, xOffset=0, yOffset=0, magInternal=10,
                           fn_out=None):
    # width and height are in pixels

    width *= magInternal
    height *= magInternal

    Narray = (2 * math.ceil(padFac/2*(np.max([width, height])))) + 1  # Make odd

    # Coordinates (at low resolution)
    xs = np.linspace(-(Narray-1)/2, (Narray-1)/2, Narray)
    ys = xs
    [XS, YS] = np.meshgrid(xs, ys)
    RS2 = XS**2 + YS**2
    RS = np.sqrt(RS2)

    if width > height:

        roc = height/2
        width_new = width - 2*roc
        height_new = height 

        xc_circle_0 = -(width/2 - roc)
        yc_circle_0 = 0

        xc_circle_1 = width/2 - roc
        yc_circle_1 = 0

    elif height > width:

        roc = width/2
        width_new = width 
        height_new = height - 2*roc

        xc_circle_0 = 0
        yc_circle_0 = -(height/2 - roc)

        xc_circle_1 = 0
        yc_circle_1 = height/2 - roc

    else:
        raise ValueError('Width and height cannot be equal for this function.')

    maskInnerRect = np.zeros_like(XS)
    maskInnerRect = (
        (XS >= -width_new/2) &
        (XS <= width_new/2) &
        (YS >= -height_new/2) &  
        (YS <= height_new/2)
        )

    maskSharpBinNegative = np.ones_like(RS)

    maskSharpBinNegative = (
        ((XS-xc_circle_0)**2 + (YS-yc_circle_0)**2 > roc**2) &
        ((XS-xc_circle_1)**2 + (YS-yc_circle_1)**2 > roc**2) &
        ~maskInnerRect
        )

    maskSharpBin = ~maskSharpBinNegative

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

        subpixel = 0*subpixel;

        Xup = Xup0 + XS[row, col]
        Yup = Yup0 + YS[row, col]
        RHOSup0 = np.sqrt((Xup-xc_circle_0)**2 + (Yup-yc_circle_0)**2)
        RHOSup1 = np.sqrt((Xup-xc_circle_1)**2 + (Yup-yc_circle_1)**2)
        # RHOSupFilletTopOuter = np.sqrt((Xup-xfcto)**2 + (Yup-yfcto)**2)
        # RHOSupFilletBottomOuter = np.sqrt((Xup-xfcbo)**2 + (Yup-yfcbo)**2)
        # RHOSupFilletTopInner = np.sqrt((Xup-xfcti)**2 + (Yup-yfcti)**2)
        # RHOSupFilletBottomInner = np.sqrt((Xup-xfcbi)**2 + (Yup-yfcbi)**2)

        # maskInnerRectUp = np.zeros_like(Xup)
        maskInnerRectUp = (
            (Xup >= -width_new/2) &
            (Xup <= width_new/2) &
            (Yup >= -height_new/2) &  
            (Yup <= height_new/2)
            )

        subpixelNegative = (
            (RHOSup0 > roc**2) &
            (RHOSup1 > roc**2) &
            ~maskInnerRectUp
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

    # %% Plotting and output writing

    # plt.figure(12)
    # plt.imshow(maskRoundedGray)
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.pause(1e-2)

    # plt.figure(11)
    # plt.imshow(maskOut)
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # plt.pause(1e-2)

    # plt.show()

    if fn_out is not None:
        fits.writeto(fn_out, maskOut, overwrite=True)

    return maskOut


if __name__ == '__main__':

    # Make sure to match EXCAM orientation. In this case can just rotate 90 degrees.
    gen_fully_rounded_slit(w_fwhm_slit, h_fwhm_slit, magInternal=10, rotDeg=90,
                           fn_out=os.path.join(HERE, 'fsam_r1c2_excam_res.fits'))
