# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=line-too-long
"""
Functions to build a relative DM probe and other propagation manipulation
"""

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.dmshapes import probe
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check
from howfsc.util.prop_tools import efield, open_efield


def make_dmrel_probe_gaussian(cfg, dmlist, dact, xcenter, ycenter, sigma, target, lod_min, lod_max,
                              ind, maxiter=5, verbose=True):
    """
    Make a relative DM probe setting whose probe height is equal to an input.

    This function creates a Gaussian probe, and adjusts the height of that
    probe iteratively until |probe amplitude|**2 within a user-specified region
    approaches a user-specified target.

    User region is annular, from lod_min to lod_max

    If using verbose: the first iteration should be off the 'target' input by
    no more than a factor of low O(1).  If off by 1-2 orders of magnitude from
    target, check the xcenter and ycenter and verify that the probe peak is
    not blocked by a pupil obscuration.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.
     dact: diameter of pupil, in actuators. > 0.
     xcenter: number of actuators to move the center of the DM pattern along
      the positive x-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     ycenter: number of actuators to move the center of the DM pattern along
      the positive y-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     sigma: width of Gaussian probe, in actuators.  > 0.
     target: desired probe intensity (i.e. |probe amplitude|**2) within the
      focal plane region of interest).  > 0.
     lod_min: minimum L/D for region of interest, must be less than lod_max.
      >= 0.
     lod_max: maximum L/D for region of interest.  > 0.
     ind: index of cfg wavelength to use for model

    Keyword Arguments:
     maxiter: number of times to iterate on DM setting.  integer > 0.
      Defaults to 5.
     verbose: if True, prints status to command line.  Defaults to True.

    Returns:
     tuple with:
      - DM1 relative DM setting, in volts
      - probe normalized-intensity 2D array in the focal plane
      - boolean mask used to select pixels to evaluate in the focal plane
      - a map of the DM surface in radians in the pupil plane
      - the product of the amplitudes of 'epup', 'pupil', and 'lyot', which
        show the region of physical obscuration in the pupil plane.  Note if
        these masks are different sizes, the output along each dimension will
        be the largest of the three on that dimension.
      The DM surface map in radians and the product of amplitudes will have the
      same array size.

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_scalar(xcenter, 'xcenter', TypeError)
    check.real_scalar(ycenter, 'ycenter', TypeError)
    check.real_positive_scalar(target, 'target', TypeError)
    check.real_nonnegative_scalar(lod_min, 'lod_min', TypeError)
    check.real_positive_scalar(lod_max, 'lod_max', TypeError)
    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if lod_min >= lod_max:
        raise ValueError('lod_min must be strictly less than lod_max')
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    check.positive_scalar_integer(maxiter, 'maxiter', TypeError)
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')

    scale = 1 # first guess: assume we got the amplitude right
    dind = 0 # only using DM1 to probe

    iopen = np.abs(open_efield(cfg, dmlist, ind))**2
    ipeak = np.max(iopen)

    ppl = cfg.sl_list[ind].dh.pixperlod
    nrow, ncol = cfg.sl_list[ind].dh.e.shape
    rld, cld = np.meshgrid((np.arange(nrow) - nrow//2)/ppl,
                           (np.arange(ncol) - ncol//2)/ppl,
                           indexing='ij')
    lod_mask = np.logical_and(np.sqrt(rld**2 + cld**2) >= lod_min,
                          np.sqrt(rld**2 + cld**2) <= lod_max)
    lod_mask = np.logical_and(lod_mask,
                              cfg.sl_list[ind].dh.e) # only keep valid pix

    j = 0
    measph = target # no effect on first iteration
    while j < maxiter:
        scale = scale / measph * target

        # Since we're iterating to get the probe amplitude right, don't bother
        # trying to account for any other scalar factors
        dp0 = probe_gaussian(cfg.dmlist[dind].registration['nact'],
                             dact,
                             xcenter,
                             ycenter,
                             sigma,
                             np.sqrt(scale)
                             )
        dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp0, cfg.sl_list[ind].lam)

        eplus = efield(cfg, [dmlist[0]+dpv, dmlist[1]], ind)
        eminus = efield(cfg, [dmlist[0]-dpv, dmlist[1]], ind)

        pampe = np.abs((eplus - eminus) / 2j)
        probe_int = pampe**2 / ipeak

        measph = np.mean(probe_int[lod_mask])
        if verbose:
            print("iteration " + str(j) + ": measured = " + str(measph) +
                  ", fractional error: " + str((measph - target) / target))

        j += 1
        pass

    # Redo with final scale
    dp0 = probe_gaussian(cfg.dmlist[dind].registration['nact'],
                         dact,
                         xcenter,
                         ycenter,
                         sigma,
                         np.sqrt(scale)
                         )

    dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp0, cfg.sl_list[ind].lam)

    # create additional data products to help users
    epups = cfg.sl_list[ind].epup.e.shape
    sps = cfg.sl_list[ind].pupil.e.shape
    lys = cfg.sl_list[ind].lyot.e.shape
    nrow = max(epups[0], sps[0], lys[0])
    ncol = max(epups[1], sps[1], lys[1])

    dm_surface = dmhtoph(
        nrow=nrow,
        ncol=ncol,
        dmin=dpv,
        nact=cfg.dmlist[dind].registration['nact'],
        inf_func=cfg.dmlist[dind].registration['inf_func'],
        ppact_d=cfg.dmlist[dind].registration['ppact_d'],
        ppact_cx=cfg.dmlist[dind].registration['ppact_cx'],
        ppact_cy=cfg.dmlist[dind].registration['ppact_cy'],
        dx=cfg.dmlist[dind].registration['dx'],
        dy=cfg.dmlist[dind].registration['dy'],
        thact=cfg.dmlist[dind].registration['thact'],
        flipx=cfg.dmlist[dind].registration['flipx'],
    )

    pupil_mask = (
        insertinto(np.abs(cfg.sl_list[ind].epup.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].pupil.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].lyot.e), (nrow, ncol))
    )

    return dpv, probe_int, lod_mask, dm_surface, pupil_mask


def probe_gaussian(nact, dact, xcenter, ycenter, sigma, height):
    """
    Create a Gaussian probe.

    Nominal DM centration with even-numbered DM counts places the center of the
     DM (and thus the center of the probe pattern) at the gap between the two
     central actuators in both axes.  The centers of the adjacent actuators are
     at (+/- 0.5 act, +/- 0.5 act).

    The probe pattern is normalized such that the Fourier transform of the
    probe pattern has amplitude "height".

    Arguments:
     nact: number of actuators along one side of the DM (assumes square DM)
     dact: diameter of pupil, in actuators
     xcenter: number of actuators to move the center of the DM pattern along
      the positive x-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     ycenter: number of actuators to move the center of the DM pattern along
      the positive y-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     sigma: width of Gaussian probe, in actuators.  > 0.
     height: height of sinc peak, in meters; actual shape may not reach this
      value depending on centration

    Returns:
     a nact x nact 2D array of heights in meters for each actuator
    """

    # Check inputs
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_scalar(xcenter, 'xcenter', TypeError)
    check.real_scalar(ycenter, 'ycenter', TypeError)
    check.real_positive_scalar(height, 'height', TypeError)

    # Set up grids with translation
    xx, yy = np.meshgrid(np.arange(nact)-(nact-1.)/2.-xcenter,
                           np.arange(nact)-(nact-1.)/2.-ycenter)

    ddm = height * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return ddm
