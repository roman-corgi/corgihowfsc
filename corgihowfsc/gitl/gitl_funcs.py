# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Helper functions for gitl loop
"""

import logging
import os

import numpy as np

import eetc
from eetc.cgi_eetc import CGIEETC
from eetc.thpt_tools import ThptToolsException
from eetc.excam_tools import EXCAMOptimizeException

from howfsc.status_codes import status_codes

from howfsc.sensing.preclean import extract_bp, normalize, eval_c
from howfsc.sensing.probephase import probe_ap
from howfsc.sensing.pairwise_sensing import estimate_efield

from howfsc.control.cs import ControlStrategy, get_wdm, get_we0
from howfsc.control.parse_cs import CSException
from howfsc.control.inversion import jac_solve, inv_to_dm, InversionException
from howfsc.control.nextiter import get_next_c, expected_time, \
     get_scale_factor_list
from howfsc.control.calcjacs import CalcJacsException
from howfsc.control.calcjtwj import JTWJMap

from howfsc.model.mode import CoronagraphMode
from howfsc.model.singlelambda import SingleLambdaException
from howfsc.model.parse_mdf import MDFException

from howfsc.util.insertinto import insertinto
from howfsc.util.gitl_tools import validate_dict_keys, param_order_to_list, \
    remove_subnormals, as_f32_normal
import howfsc.util.check as check
from howfsc.util.constrain_dm import ConstrainDMException
from howfsc.util.actlimits import ActLimitException
from howfsc.util.check import CheckException

log = logging.getLogger(__name__)

toplevel_keys = ['overhead', 'star', 'excam', 'hardware', 'howfsc', 'probe']
overhead_keys = ['overdm', 'overfilt', 'overboth', 'overfixed', 'overframe']
star_keys = ['stellar_vmag', 'stellar_type',
             'stellar_vmag_target', 'stellar_type_target']
excam_keys = ['cleanrow', 'cleancol', 'scale_method', 'scale_percentile',
              'scale_bright_method', 'scale_bright_percentile']
hardware_keys = ['sequence_list', 'sequence_observation', 'pointer']
howfsc_keys = ['method', 'min_good_probes', 'eestclip', 'eestcondlim']
probe_keys = ['dmrel_ph_list']

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))

def check_inputs(framelist, dm1_list, dm2_list, cfg, jac, jtwj_map,
                             croplist, prev_exptime_list,
                             cstrat, n2clist, hconf, iteration):
    """

    Check inputs of howfsc

    For reference in defined the below arguments, the number of CFAM filters
    used is nlam, and the number of DM settings used is ndm.  ndm will be
    odd, as the expectation of this sensing approach is that we use nprobepair
    pairs of DM settings (one + relative to base, one -) and one base setting
    with no probes.

    Arguments:
     framelist: nlam*ndm list of HOWFSC GITL frames.  These are nrow x ncol
      intensity images as floating-point arrays, which have already been
      corrected for bias, dark and flat, and converted into units of photons.
      The order should match the order in which they are collected, which is
      to cycle the wavelengths as the outer loop, and the DM settings as the
      inner loop, starting with the unprobed image.  Examplw ordering:

      [lam0 unprobed, lam0 probe0+, lam0 probe0-, lam0 probe1+, lam0 probe1-,
       lam0, probe2+, lam0 probe2-, lam1 unprobed, lam1 probe0+, ...]

       framelist data will be sourced from HOWFSC packets for GITL.

     dm1_list: nlam*ndm list of DM1 settings.  These are 48x48 floating-point
      arrays with absolute DM settings in volts.  Each list element corresponds
      to the same data collection as framelist; they are in the same order.
      dm1_list data will be sourced from HOWFSC packets for DM1 settings.

      Under the current probing assumptions, the ndm settings used for DM1 will
      vary with a given CFAM filter, but when moving to the next filter, the
      same ndm settings will be repeated, modulo any small changes due to
      localized neighbor rule violations when running in conjunction with the
      Zernike loop.

     dm2_list: nlam*ndm list of DM2 settings.  These are 48x48 floating-point
      arrays with absolute DM settings in volts.  Each list element corresponds
      to the same data collection as framelist; they are in the same order.
      dm2_list data will be sourced from HOWFSC packets for DM2 settings.

      Under the current probing assumptions, dm2_list elements will all be the
      same.

     cfg: a CoronagraphMode object (i.e. optical model)

     jac: 3D real-valued DM Jacobian array, as produced by calcjacs().
      Shape is 2 x ndm x npix.

     jtwj_map: a JTWJMap object which collects precalculated
      jac.T * diag(we0)**2 * jac matrices for all of the weighting matrices in
      the control strategy; each of these includes all fixed
      bad pixels and all per-pixel weighting, but does not include any bad
      pixels that vary (e.g. from cosmic ray flux).  Can use internal methods
      to return an appropriate jtwj matrix, which should be a 2D ndm x ndm
      array.

     croplist: list of 4-tuples of (lower row, lower col,
      number of rows, number of columns), indicating where in a clean frame
      each PSF is taken.  All are integers; the first two must be >= 0 and the
      second two must be > 0.  This should have ndm*nlam elements, and elements
      corresponding to the same wavelengths should have the same crop settings.
      croplist data will be sourced from HOWFSC packets for ancillary GITL
      info.

     prev_exptime_list: list of exposure times for each of the frames in
      framelist.  framelist data is averaged on board, so this will be the
      EXCAM exposure for a single frame during the data collection period that
      fed that frame.  This should have ndm*nlam elements.  prev_exptime_list
      data will be sources from HOWFSC packets for ancillary GITL info.

     cstrat: a ControlStrategy object; this will be used to define the behavior
      of the wavefront control by setting the regularization, per-pixel
      weighting, multiplicative gain, and next-iteration probe height.  It will
      also contain information about fixed bad pixels.

     n2clist: list of 2D floating-point arrays giving the scale factor to
      convert from normalized intensity to contrast as a multiplier.  These
      correspond to the relative drop-off in flux from unity due to the
      presence of nearby coronagraphic masks.

      As an example, if the measured normalized intensity is 5e-9, but the
      presence of a mask edge is causing the flux to drop to 0.5x of usual,
      then the value in an array of n2clist will be 2, and the actual contrast
      will be 1e-8.

      Elements of n2clist must be at least as large as the number of rows and
      columns in a cropped region (elements 2 and 3 of each croplist tuple) to
      ensure that every valid pixel has a conversion factor to contrast.  This
      should have the same number of elements as the model has wavelengths.

     hconf: dictionary of dictionaries.  Keys at the top level must match the
      set in toplevel_keys, while each subdictionary must have its own keys
      match the relevant list.  These contain scalar configuration parameters
      which do not change iteration-to-iteration.

     iteration: integer > 0 giving the number of the iteration which is about
      to happen.  Iteration 0 is the setup which CGI is initialized with at
      startup; iteration 1 is the first iteration calculated by HOWFSC GITL,
      and the data collection for that iteration follows the first calculation.

    Returns:
     - An absolute DM setting for DM1
     - An absolute DM setting for DM2
     - 6 scalar floating-point probe height scale values for the three
      relative-DM probes, stored in a list.  First half are positive values,
      second half is the negative of those values.
     - An EXCAM gain for the absolute DM setting and each pair of relative
      probe settings, for each CFAM filter.  This is stored as a list of lists,
      where the outer list has nlam elements, and the inner has (1 + the number
      of probe pairs).  This breakdown is chosen to match the CGI parameter set
     - An EXCAM exposure time for the absolute DM setting and each pair of
      relative probe settings, for each CFAM filter. This is stored as a list
      of lists, where the outer list has nlam elements, and the inner has
      (1 + the number of probe pairs).  This breakdown is chosen to match the
      CGI parameter set
     - A number of EXCAM frames for the absolute DM setting and each pair of
      relative probe settings, for each CFAM filter. This is stored as a list
      of lists, where the outer list has nlam elements, and the inner has
      (1 + the number of probe pairs).  This breakdown is chosen to match the
      CGI parameter set
     - The mean total contrast measured from the previous iteration (the one
      which provided the input data)
     - The mean total contrast expected for the next iteration based on the
      absolute DM settings
     - The expected time to complete the next iteration
     - A status code which = 0 if the computation completed without errors,
      and a nonzero code if the computation failed, with the value indicating
      the cause.
     - a final catch-all dictionary which contains data products useful for
      understanding and monitoring HOWFSC performance but not necessary for
      preparing the next HOWFSC iteration.

    """

    #--------------
    # Check inputs
    #--------------

    log.info('Begin input checks')

    # iteration
    check.positive_scalar_integer(iteration, 'iteration', TypeError)
    log.info('Iteration number: %d', iteration)

    # cfg first as we need it immediately
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    # framelist
    try:
        lenflist = len(framelist)
    except TypeError: # not iterable
        raise TypeError('framelist must be an iterable') # reraise
    for index, gitlframe in enumerate(framelist):
        check.twoD_array(gitlframe, 'framelist[' + str(index) + ']', TypeError)
        pass

    nlam = len(cfg.sl_list)
    ndm = lenflist // nlam
    if lenflist % ndm != 0: # did not divide evenly
        raise TypeError('Number of received frames not consistent with ' +
                        'number of model wavelengths and assumption of ' +
                        'identical probing per wavelength')
    if ndm % 2 != 1: # expect an odd number of DMs: N pairs + 1 unprobed
        raise TypeError('Expected odd number of DM1 settings, got even')
    nprobepair = (ndm - 1)//2 # will be int since ndm odd

    # dm1_list
    try:
        lendm1list = len(dm1_list)
    except TypeError: # not iterable
        raise TypeError('dm1_list must be an iterable') # reraise
    dm1nact = cfg.dmlist[0].registration['nact']
    for index, dm1 in enumerate(dm1_list):
        check.twoD_array(dm1, 'dm1_list[' + str(index) + ']', TypeError)
        if dm1.shape != (dm1nact, dm1nact):
            raise TypeError('Unexpected dm1 shape ' + str(dm1.shape) +
                            ' at index ' + str(index) + ' of dm1_list')
        pass
    if lendm1list != lenflist:
        raise TypeError('Number of frames and number of DM1 settings must ' +
                        'be the same')

    # dm2_list
    try:
        lendm2list = len(dm2_list)
    except TypeError: # not iterable
        raise TypeError('dm2_list must be an iterable') # reraise
    dm2nact = cfg.dmlist[1].registration['nact']
    for index, dm2 in enumerate(dm2_list):
        check.twoD_array(dm2, 'dm2_list[' + str(index) + ']', TypeError)
        if dm2.shape != (dm2nact, dm2nact):
            raise TypeError('Unexpected dm2 shape ' + str(dm2.shape) +
                            ' at index ' + str(index) + ' of dm2_list')
        pass
    if lendm2list != lenflist:
        raise TypeError('Number of frames and number of DM2 settings must ' +
                        'be the same')

    # jac + jtwj_map
    check.threeD_array(jac, 'jac', TypeError) # axis 0 is real/imag
    if not isinstance(jtwj_map, JTWJMap):
        raise TypeError('jtwj_map must be a JTWJMap object')

    if jac.shape[0] != 2:
        raise TypeError('jac axis 0 must be length 2 (real/imag)')

    allpix = np.sum([np.sum(sl.dh.e) for sl in cfg.sl_list])
    if jac.shape[2] != allpix:
        raise TypeError('jac and cfg have inconsistent number of dark-hole ' +
                        'pixels: jac = ' + str(jac.shape[2]) + ', cfg = ' +
                        str(allpix))

    # croplist
    try:
        lencroplist = len(croplist)
    except TypeError: # not iterable
        raise TypeError('croplist must be an iterable') # reraise
    for index, crop in enumerate(croplist):
        if not isinstance(crop, tuple):
            raise TypeError('croplist[' + str(index) + '] must be a tuple')
        if len(crop) != 4:
            raise TypeError('Each element of croplist must be a 4-tuple')
        check.nonnegative_scalar_integer(crop[0], 'croplist[' +
                                         str(index) + '][0]', TypeError)
        check.nonnegative_scalar_integer(crop[1], 'croplist[' +
                                         str(index) + '][1]', TypeError)
        check.positive_scalar_integer(crop[2], 'croplist[' +
                                      str(index) + '][2]', TypeError)
        check.positive_scalar_integer(crop[3], 'croplist[' +
                                      str(index) + '][3]', TypeError)
        pass
    if lencroplist != len(framelist):
        raise TypeError('croplist and framelist must contain the same ' +
                        'number of elements')

    nrow = croplist[0][2] # array size
    ncol = croplist[0][3]
    for index, _ in enumerate(croplist):
        if croplist[index][2] != nrow:
            raise ValueError('Not all nrow values in incoming data are ' +
                             'identical; suggests data corruption')
        if croplist[index][3] != ncol:
            raise ValueError('Not all ncol values in incoming data are ' +
                             'identical; suggests data corruption')
        pass
    for f in framelist:
        if f.shape != (nrow, ncol):
            raise TypeError('Row/col size data does not match image shape')
        pass

    subcroplist = [] # crop data should only vary with wavelength
    for index, _ in enumerate(cfg.sl_list):
        subcroplist.append(croplist[index*ndm])
        for j in range(1, ndm): # no variation with DM setting
            if croplist[index*ndm] != croplist[index*ndm + j]:
                raise ValueError('Crop data not identical across DM ' +
                                 'changes; suggests data corruption')
            pass
        pass

    # prev_exptime_list
    try:
        lenpelist = len(prev_exptime_list)
    except TypeError: # not iterable
        raise TypeError('prev_exptime_list must be an iterable') # reraise
    for index, prev in enumerate(prev_exptime_list):
        check.real_positive_scalar(prev, 'prev_exptime_list[' + str(index) +
                                   ']', TypeError)
        pass
    if lenpelist != lenflist:
        raise TypeError('prev_exptime_list and framelist must contain the ' +
                        'same number of elements')

    # cstrat
    if not isinstance(cstrat, ControlStrategy):
        raise TypeError('cstrat must be a ControlStrategy object')

    # n2clist
    if not isinstance(n2clist, list):
        raise TypeError('n2clist must be a list')
    if len(n2clist) != len(cfg.sl_list):
        raise TypeError('Number of NI-to-contrast conversion matrices '+
                        'does not match model')
    for index, n2c in enumerate(n2clist):
        check.twoD_array(n2c, 'n2clist[' + str(index) + ']', TypeError)
        if n2c.shape[0] < subcroplist[index][2]:
            raise TypeError('Number of rows in ' +
                            'n2clist[' + str(index) + '] must be at least ' +
                            'as large as croplist[' + str(index) + '][2]')
        if n2c.shape[1] < subcroplist[index][3]:
            raise TypeError('Number of columns in ' +
                            'n2clist[' + str(index) + '] must be at least ' +
                            'as large as croplist[' + str(index) + '][3]')
        pass

    # hconf
    validate_dict_keys(hconf, toplevel_keys, custom_exception=TypeError)
    validate_dict_keys(hconf['overhead'], overhead_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['star'], star_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['excam'], excam_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['hardware'], hardware_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['howfsc'], howfsc_keys,
                       custom_exception=TypeError)
    validate_dict_keys(hconf['probe'], probe_keys,
                       custom_exception=TypeError)

    # Compute/check against derived parameters
    if len(hconf['probe']['dmrel_ph_list']) != nprobepair:
        raise TypeError('Number of provided DM probe heights in ' +
                        'dmrel_ph_list not consistent with number of probe ' +
                        'pairs derived from dm1_list')

    if len(n2clist) != nlam:
        raise TypeError('Too few NI-to-contrast conversion matrices')
    if len(hconf['hardware']['sequence_list']) != nlam:
        raise TypeError('Number of provided seqs in sequence_list not ' +
                        'consistent with number of model wavelengths')

    log.info('Input checks complete')

    return lenflist, nlam, ndm, nprobepair, lendm1list, lendm2list, dm1nact, dm2nact, allpix, lencroplist, nrow, ncol, subcroplist, lenpelist

