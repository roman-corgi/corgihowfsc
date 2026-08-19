# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Generate SPM array with fiducials for SPAM calibration."""
import os
import numpy as np
from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt

from cal.util.insertinto import insertinto
from cal.util.loadyaml import loadyaml
from cal.util import shapes
from cal.pupilfit import pupilfit_gsw
import cal.maskgen.maskgen as mg

MASKGEN_PATH = pathlib.Path(mg.__file__).resolve().parent
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


# %% Load data

fnShapedPupilCalib = os.path.join(IN_PATH, 'any_band', 'params_fit_spam_mag_clocking.yaml')
# fnShapedPupilCalib = os.path.join(PUPILFIT_PATH, 'cgidata',
#                                   'fit_spam_mag_clocking.yaml')

inp = loadyaml(fnShapedPupilCalib)
nIter = inp['nIter']
clockMaxDeg = inp['clockMaxDeg']
nClock = inp['nClock']
DeltaMag = inp['DeltaMag']
nMag = inp['nMag']
dBeam0 = inp['dBeamNom']
dCircleM = inp['dCircleM']
dyCirc = inp['dyCircM']
# nBeam0 = inp['nBeamNom']
nPixFFT = inp['nPixFFT']
nPixOut = inp['nPixOut']
nSubpixels = inp['nSubpixels']

aoiRad = np.radians(7.5)  # radians
dxSPM = 9.3e-3 * np.cos(aoiRad)  # [meters]. Valid at small rotations

# %% USER-DEFINED INPUTS

flagPlot = False
flagWrite = True

# nBeam0 = 309  # For HLC mode in CGI PROPER model
# nArray = 401  # Used only when generating the image

nBeam0 = 1000  # For HLC mode in CGI PROPER model
nArray = 1101  # Used only when generating the image

# %% INPUTS FOR GENERATING TEST DATA

# (Leave xc and yc alone)
xc = 0.0  # [meters]
yc = 0.0  # [meters]

xcImagePix = 0
ycImagePix = 0
imageShape = (nArray, nArray)

magList = [1, ] #[0.95, 0.97, 0.99, 1.01, 1.03, 1.05]
rotList = [0,] # [-2, -1, 0, 1, 2]  # degrees

# # Single case for debugging
# magList = [0.95, ]
# rotList = [-2, ]  # degrees

for iMag, mag in enumerate(magList):

    for iRot, rot in enumerate(rotList):

        print('Generating SPAM calibration mask for:    ' +
              ('mag = %.3f, rot = %.3fdeg' % (mag, rot)))

        clockAllDeg = rot
        clockAllRad = clockAllDeg*(np.pi/180.0)
        nBeam = nBeam0*mag
        dx = dBeam0 / nBeam  # meters per pixel at output

        amp = np.zeros((nArray, nArray))

        # Read in the SPMs in the top row
        spmLeft = fits.getdata(
            os.path.join(
                MASKGEN_PATH, 'maskdesigns', 'SPC_20200610_WFOV',
                'SPM_SPC_20200610_WFOV_1000us3x_binary_uint8.fits'))
        spmRight = fits.getdata(
            os.path.join(
                MASKGEN_PATH, 'maskdesigns', 'SPC_20200628_Spec_Rot60',
                'SPM_SPC_20200628_Spec_Rot60_1000us3x_binary_uint8.fits'))
        # # Read in the pupil
        # pupil = fits.getdata(
        #     os.path.join(PUPILFIT_PATH, 'cgidata',
        #                  'pupil_template_D2000.00pixels.fits'))

        xcLeft = -np.cos(clockAllRad)*dxSPM   # x center of left SPM [meters]
        ycLeft = -np.sin(clockAllRad)*dxSPM   # y center of left SPM [meters]
        xcRight = np.cos(clockAllRad)*dxSPM   # x center of left SPM [meters]
        ycRight = np.sin(clockAllRad)*dxSPM   # y center of left SPM [meters]

        # Resize and translate the SPMs and pupil
        spmLeftPad = insertinto(spmLeft, (3200, 3200))
        spmRightPad = insertinto(spmRight, (3200, 3200))
        # pupilPad = insertinto(pupil, (2200, 2200))

        spmLeftMoved = mg.rotate_shift_downsample_amplitude_mask(
            spmLeftPad, clockAllDeg, nBeam/3000, (xc+xcLeft)/dx, (yc+ycLeft)/dx)

        spmRightMoved = mg.rotate_shift_downsample_amplitude_mask(
            spmRightPad, 60+clockAllDeg, nBeam/3000, (xc+xcRight)/dx, (yc+ycRight)/dx)

        # pupilMoved = mg.rotate_shift_downsample_amplitude_mask(
        #     pupilPad, 0, nBeam/2000, xc/dx, yc/dx)

        spmLeftMoved = insertinto(spmLeftMoved, (nArray, nArray))
        spmRightMoved = insertinto(spmRightMoved, (nArray, nArray))
        # pupilMoved = insertinto(pupilMoved, (nArray, nArray))

        amp += (spmLeftMoved + spmRightMoved)  # * pupilMoved

        spms = spmLeftMoved + spmRightMoved

        # Generate the alignment circles
        circ1 = shapes.circle(
            nArray, nArray, dCircleM / dx/2.0,
            (xc - np.sin(clockAllRad)*dyCirc/2)/dx,
            (yc + np.cos(clockAllRad)*dyCirc/2)/dx,
            nSubpixels=100)
        circ2 = shapes.circle(
            nArray, nArray, dCircleM / dx/2.0,
            (xc + np.sin(clockAllRad)*dyCirc/2)/dx,
            (yc - np.cos(clockAllRad)*dyCirc/2)/dx,
            nSubpixels=100)
        amp += circ1 + circ2
        spotMeas = np.abs(amp)**2

        # De-center the generated pupil image
        spotMeas = insertinto(spotMeas, imageShape)
        spotMeas = np.rot90(spotMeas, 2)
        # spotMeas = np.roll(spotMeas, (ycImagePix, xcImagePix), axis=(0, 1))

        if flagPlot:
            plt.figure(1)
            plt.imshow(spotMeas)
            plt.set_cmap('gray')
            plt.gca().invert_yaxis()
            plt.pause(1e-2)

        # Write to file
        hdu = fits.PrimaryHDU(spotMeas)
        fnTestdata = os.path.join(
            OUT_PATH,
            'spam',
            'spm_calib_spots_n%d_mag%.3f_clk%.3fdeg.fits' % (nBeam0, mag, clockAllDeg)
        )
        hdu.writeto(fnTestdata, overwrite=True)
