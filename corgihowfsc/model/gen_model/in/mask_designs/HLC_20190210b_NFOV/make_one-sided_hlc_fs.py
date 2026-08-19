"""
Make a gray-edged, half-annulus field stop for the Band 1 HLC.

Steps:
1. Create a binary version of the mask
2. Find all the edge pixels
3. Upsample at all the edge pixels to compute the fractional value at each.

@author: ajriggs
"""
import math
import os

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))

# %% User-defined parameters

res = 25  # pixels per lambda/D
halfWidth = 10
Narray = 2*math.ceil(res*halfWidth)+1
upsampleFactor = 100

Rin = 2.3
Rout = 9.7
Rf = 41.626*0.575/30


# %%

# Coordinates (at low resolution)
xs = np.linspace(-(Narray-1)/2, (Narray-1)/2, Narray)/res
ys = xs
[XS, YS] = np.meshgrid(xs, ys)
RS2 = XS**2 + YS**2
RS = np.sqrt(RS2)

# Fillet center (top, outer)
xfcto = Rf
yfcto = np.sqrt(Rout**2 - 2*Rout*Rf)
mfcto = yfcto/xfcto

# Fillet center (bottom, outer)
xfcbo = Rf
yfcbo = -yfcto
mfcbo = yfcbo/xfcbo

# Fillet center (top, inner)
xfcti = Rf
yfcti = np.sqrt(Rin**2 + 2*Rin*Rf)
mfcti = yfcti/xfcti

# Fillet center (bottom, inner)
xfcbi = Rf
yfcbi = -yfcti
mfcbi = yfcbi/xfcbi

# # Fillet intersection with OD (top, outer)
# xfito = Rout / np.sqrt(1 + mfcto**2)
# yfito = mfcto * xfito

maskSharpBin = np.zeros_like(RS)

maskSharpBin = (XS >=0) & (RS <= Rout) & (RS>= Rin)

filletTopOuter = (((XS - xfcto)**2 + (YS - yfcto)**2 > Rf**2) &
             (YS > yfcto) &
             (YS > mfcto*XS) &
             maskSharpBin
            )

filletBottomOuter = (((XS - xfcbo)**2 + (YS - yfcbo)**2 > Rf**2) &
             (YS < yfcbo) &
             (YS < mfcbo*XS) &
             maskSharpBin
            )

filletTopInner = (((XS - xfcti)**2 + (YS - yfcti)**2 > Rf**2) &
             (YS < yfcti) &
             (YS > mfcti*XS) &
             maskSharpBin
            )

filletBottomInner = (((XS - xfcbi)**2 + (YS - yfcbi)**2 > Rf**2) &
             (YS > yfcbi) &
             (YS < mfcbi*XS) &
             maskSharpBin
            )

fillets = filletTopOuter | filletBottomOuter | filletTopInner | filletBottomInner

maskRoundedBin =  maskSharpBin & ~fillets

maskSharp = maskSharpBin.astype(float)
maskRounded = maskRoundedBin.astype(float)

kernel = np.ones((3, 3))

# filletsTemp = convolve2d(fillets, kernel, mode='same') / np.sum(kernel)
# # grayIndsFillet = np.nonzero(np.logical_and(filletsTemp > 0, filletsTemp < 1))
# grayIndsFillet = np.nonzero(filletsTemp > 1/(2*np.sum(kernel)))
# filletEdges = np.zeros_like(RS)
# filletEdges[grayIndsFillet] = True

roundedTemp = convolve2d(maskRounded, kernel, mode='same') / np.sum(kernel)
grayIndsRounded = np.nonzero(np.logical_and(roundedTemp > 0, roundedTemp < 1))
roundedEdges = np.zeros_like(RS)
roundedEdges[grayIndsRounded] = True

annulusTemp = convolve2d(maskSharp, kernel, mode='same') / np.sum(kernel)
grayIndsAnnulus = np.nonzero(np.logical_and(annulusTemp > 0, annulusTemp < 1))
annularEdges = np.zeros_like(RS)
annularEdges[grayIndsAnnulus] = True


# %% Upsampling

dx = 1/res
dxUp = dx/upsampleFactor;
xUp = np.linspace(-(upsampleFactor-1)/2, (upsampleFactor-1)/2, upsampleFactor)*dxUp
[Xup0, Yup0] = np.meshgrid(xUp, xUp)

subpixel = np.zeros((upsampleFactor, upsampleFactor))
allFillets = np.zeros((Narray, Narray))
allShapes = 1 - maskRounded.copy()
annulusGray = maskSharp.copy()

# Make the grayscale versions of the fillets
for ii in range(len(grayIndsRounded[0])):
    
    row = grayIndsRounded[0][ii]
    col = grayIndsRounded[1][ii]
    
    subpixel = 0*subpixel;

    Xup = Xup0 + XS[row, col]
    Yup = Yup0 + YS[row, col]
    RHOSup = np.sqrt(Xup**2 + Yup**2)
    RHOSupFilletTopOuter = np.sqrt((Xup-xfcto)**2 + (Yup-yfcto)**2)
    RHOSupFilletBottomOuter = np.sqrt((Xup-xfcbo)**2 + (Yup-yfcbo)**2)
    RHOSupFilletTopInner = np.sqrt((Xup-xfcti)**2 + (Yup-yfcti)**2)
    RHOSupFilletBottomInner = np.sqrt((Xup-xfcbi)**2 + (Yup-yfcbi)**2)
    
    subpixel[
            ((RHOSupFilletTopOuter > Rf) &
            (Yup > yfcto) &
            (Yup > mfcto*Xup)) |
            
            ((RHOSupFilletBottomOuter > Rf) &
            (Yup < yfcbo) &
            (Yup < mfcbo*Xup)) |
            
            ((RHOSupFilletTopInner > Rf) &
            (Yup < yfcti) &
            (Yup > mfcti*Xup)) |
            
            ((RHOSupFilletBottomInner > Rf) &
            (Yup > yfcbi) &
            (Yup < mfcbi*Xup)) |
            
            ((Xup < 0) | (RHOSup > Rout) | (RHOSup < Rin))
            ] = 1

    pixelValue = np.sum(subpixel) / upsampleFactor**2
    allShapes[row, col] = pixelValue

maskRoundedGray = 1-allShapes

# %% Plotting and output writing

plt.figure(11)
plt.imshow(roundedEdges)
plt.gca().invert_yaxis()
plt.colorbar()
plt.pause(1e-2)

plt.figure(14)
plt.imshow(maskRoundedGray)
plt.gca().invert_yaxis()
plt.colorbar()
plt.pause(1e-2)

for irot in range(4):
    rot = irot * 90
    fn = os.path.join(HERE, ('FS_HLC_half_annulus_%dppl_rot%d.fits' % (res, rot)))

    fits.writeto(fn, np.rot90(maskRoundedGray, irot))

#hdu = fits.PrimaryHDU(maskRoundedGray)
#hdu.writeto(fn, overwrite=True)
