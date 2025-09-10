# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Runs the end-to-end HOWFSC Computation activity
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
# from howfsc.sensing.pairwise_sensing import estimate_efield

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

###

from corgihowfsc.gitl.gitl_funcs import check_inputs
###

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

def howfsc_computation(framelist, dm1_list, dm2_list, cfg, jac, jtwj_map,
                       croplist, prev_exptime_list,
                       cstrat, n2clist, hconf, iteration,
                       estrat, imager, normstrat, probing):
    """
    Wrapper for the main HOWFSC computation loop, to handle exceptions in a
    way consistent with the interface specifications (outputs indicated by a
    status code)

    Same inputs, same outputs.
    """
    try:
        log.info('-------------------------------')
        log.info('Begin howfsc_computation')
        out = _main_howfsc_computation(framelist, dm1_list, dm2_list,
                                       cfg, jac, jtwj_map,
                                       croplist, prev_exptime_list,
                                       cstrat, n2clist, hconf, iteration,
                                       estrat, imager, normstrat, probing)
        log.info('howfsc_computation main loop complete')
        return out
    # Note: while in principle _main_howfsc_computation() could output a result
    # with a status other than nominal, in practice the only non-nominal cases
    # identified so far are exceptions which would cause an abort.
    except TypeError:
        log.exception('Returning with status code %s',
                          status_codes['TypeError'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['TypeError'], None)
    except ValueError:
        log.exception('Returning with status code %s',
                          status_codes['ValueError'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['ValueError'], None)
    except ConstrainDMException:
        log.exception('Returning with status code %s',
                          status_codes['ConstrainDMException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['ConstrainDMException'], None)
    except InversionException:
        log.exception('Returning with status code %s',
                          status_codes['InversionException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['InversionException'], None)
    except ZeroDivisionError:
        log.exception('Returning with status code %s',
                          status_codes['ZeroDivisionError'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['ZeroDivisionError'], None)
    except KeyError:
        log.exception('Returning with status code %s',
                          status_codes['KeyError'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['KeyError'], None)
    except IOError:
        log.exception('Returning with status code %s',
                          status_codes['IOError'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['IOError'], None)
    except CalcJacsException:
        log.exception('Returning with status code %s',
                          status_codes['CalcJacsException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['CalcJacsException'], None)
    except CSException:
        log.exception('Returning with status code %s',
                          status_codes['CSException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['CSException'], None)
    except MDFException:
        log.exception('Returning with status code %s',
                          status_codes['MDFException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['MDFException'], None)
    except ActLimitException:
        log.exception('Returning with status code %s',
                          status_codes['ActLimitException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['ActLimitException'], None)
    except CheckException:
        log.exception('Returning with status code %s',
                          status_codes['CheckException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['CheckException'], None)
    except SingleLambdaException:
        log.exception('Returning with status code %s',
                          status_codes['SingleLambdaException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['SingleLambdaException'], None)
    except EXCAMOptimizeException:
        log.exception('Returning with status code %s',
                          status_codes['EXCAMOptimizeException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['EXCAMOptimizeException'], None)
    except ThptToolsException:
        log.exception('Returning with status code %s',
                          status_codes['ThptToolsException'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['ThptToolsException'], None)
    # handle everything else, there is no option for failed return without a
    # status code
    except Exception: # pylint: disable=broad-except
        log.exception('Returning with status code %s',
                          status_codes['Exception'])
        return (None, None, None, None, None, None, None, None, None,
                status_codes['Exception'], None)
    pass



def _main_howfsc_computation(framelist, dm1_list, dm2_list, cfg, jac, jtwj_map,
                             croplist, prev_exptime_list,
                             cstrat, n2clist, hconf, iteration,
                             estrat, imager, normstrat, probing):
    """
    Execute the HOWFSC GITL computation as defined in the HOWFSC FDD

    Perform the following tasks:
     1. Separate image data from bad pixel map for all frames
     2. Normalize each intensity frame
     3. Evaluate mean total contrast over control pixels which are not masked
      by bad pixel map
     4. Estimate complex electric fields and return fields with bad electric
      field maps
     5. Compute control strategy parameters
     6. Compute change in DM settings from electric fields
     7. Compute expected mean total contrast for next iteration
     8. Compute absolute DM settings, camera settings, probe scale factors
     9. Compute expected time to complete next iteration

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
    (lenflist, nlam, ndm, nprobepair, lendm1list, lendm2list, dm1nact, dm2nact,
     allpix, lencroplist, nrow, ncol, subcroplist, lenpelist) = check_inputs(framelist, dm1_list, dm2_list, cfg, jac,
                                                                             jtwj_map, croplist, prev_exptime_list,
                                                                             cstrat, n2clist, hconf, iteration)

    #--------------------
    # HOWFSC Computation
    #--------------------

    # This is a catch-all dictionary for HOWFSC information we want to export
    # from the call, but don't strictly need to set up the next iteration.
    # Will contain things like electric-field estimates, etc.
    other = dict()

    log.info('Building exposure-time calculator class')
    cgi_eetc = CGIEETC(mag=hconf['star']['stellar_vmag'],
                       phot='v', # only using V-band magnitudes as a standard
                       spt=hconf['star']['stellar_type'],
                       pointer_path=os.path.join(eetc_path,
                                                 hconf['hardware']['pointer']),
    )
    log.info('HOWFSC evaluated with Vmag = %g, type = %s',
             hconf['star']['stellar_vmag'],
             hconf['star']['stellar_type'],
    )

    dmlistmeas = [dm1_list[0], dm2_list[0]]
    tielist = [cfg.dmlist[0].dmvobj.tiemap, cfg.dmlist[1].dmvobj.tiemap]

    bplist = []
    dhlist = []
    unprobedlist = [] # unprobed NI for contrast estimation
    intlist = [] # NIs for estimation
    for j in range(nlam):
        other[j] = dict()
        other[j]['lam'] = cfg.sl_list[j].lam

        intlist.append(np.zeros((ndm, nrow, ncol))) # int storage for whole lam
        log.info('Wavelength %d of %d', j+1, nlam)
        log.info('Get flux for this CFAM filter')
        # _, peakflux = normalization.calc_flux_rate(
        #     sequence_name=hconf['hardware']['sequence_list'][j],
        # )
        _, peakflux = normstrat.calc_flux_rate(hconf)

        log.info('Expect %g photons/sec', peakflux)
        for k in range(ndm):
            log.info('DM setting %d of %d', k+1, ndm)
            ind = j*ndm + k
            im = framelist[ind]

            # 1. Separate image data from bad pixel map for all frames
            log.info('1. Separate image data from bad pixel map for ' +
                         'all frames')
            bplist.append(extract_bp(im))

            # 2. Normalize each intensity frame
            log.info('2. Normalize each intensity frame')
            log.info('Use peakflux = %g, exptime = %g', peakflux,
                     prev_exptime_list[ind])
            nim = normalize(im, peakflux, prev_exptime_list[ind])

            # Prep material for eval_c only on unprobed frames
            if k == 0:
                unprobedlist.append(nim)
                other[j]['meas_intensity'] = nim

                dh = cfg.sl_list[j].dh.e
                dhcrop = insertinto(dh, (nrow, ncol)).astype('bool')
                if np.sum(dh) != np.sum(dhcrop):
                    log.warning('model dark hole size cropped to fit ' +
                                    'into GITL frame size')
                    pass
                dhlist.append(dhcrop)
                pass

            # nan the pixels we aren't using anyway, so we don't both doing
            # a solve on them.  Bad pixels are pre-NaN'ed, so just do dark hole
            nim[~dhlist[-1]] = np.nan

            # Prep material for phase estimation
            intlist[j][k, :, :] = nim
            pass
        pass

    # 3. Evaluate mean total contrast over control pixels which are not masked
    # by bad pixel map
    log.info('3. Evaluate mean total contrast over control pixels which ' +
                 'are not masked by bad pixel map')
    prev_c = eval_c(unprobedlist, dhlist, n2clist)
    log.info('Previous contrast = %g', prev_c)

    # 4. Estimate complex electric fields and return fields with bad electric
    # field maps
    log.info('4. Estimate complex electric fields and return fields ' +
                 'with bad electric field maps')
    plist, other = probing.get_probe_ap(cfg, dm1_list, dm2_list, other=other)

    # plist = [] # for model-based phase storage, chunked by lam
    # for n in range(nprobepair):
    #     log.info('Probe pair %d of %d', n+1, nprobepair)
    #     # Extract phases from model
    #     # element zero is unprobed, not used here
    #     # data collection will do plus then minus
    #     for j in range(nlam):
    #         plist.append(np.zeros((nprobepair, nrow, ncol)))
    #         log.info('Wavelength %d of %d', j+1, nlam)
    #         log.info('Get probe phase from model and DM settings')
    #         _, tmpph = probe_ap(cfg,
    #                 dm1_list[j*ndm + 1+2*n], # positive
    #                 dm1_list[j*ndm + 2+2*n], # negative
    #                 dm2_list[j*ndm],
    #                 j)
    #
    #         plist[j][n, :, :] = insertinto(tmpph, (nrow, ncol))
    #
    #         # Save the probe phases for later
    #         key_n = 'probe_ph' + str(n)
    #         other[j][key_n] = np.squeeze(plist[j][n, :, :])
    #         pass
    #
    #     pass

    # Initialize accumulators for average modulated/unmodulated signal across
    # all filters for 1133642
    npixmod = 0
    modulsum = 0
    unmodulsum = 0

    # this should match dhlist rather than sl_list (though they both should
    # agree)
    log.info('Solve for electric field')
    ndhpix = np.cumsum([0]+[np.sum(dh) for dh in dhlist])
    emeas = np.zeros((ndhpix[-1],)).astype('complex128')
    bpmeas = np.zeros((ndhpix[-1],)).astype('bool')
    destlist = [] # list of measured - model 2D arrays, for next iter contrast
    for j in range(nlam):
        log.info('Wavelength %d of %d', j+1, nlam)

        # Measured e-field at this setting
        log.info('Measured e-field at this setting')
        efield = estrat.estimate_efield(
            intlist[j],
            plist[j],
            min_good_probes=hconf['howfsc']['min_good_probes'],
            eestclip=hconf['howfsc']['eestclip'],
            eestcondlim=hconf['howfsc']['eestcondlim'],
        )
        badefield = np.isnan(efield)
        efield[badefield] = 0 # nan is the one value that EFC can't fix
        emeas[ndhpix[j]:ndhpix[j+1]] = efield[dhlist[j]]
        efield[badefield] = np.nan # reset back to NaNs after EFC array filled
        bpmeas[ndhpix[j]:ndhpix[j+1]] = badefield[dhlist[j]]

        # Stash computed products for later analysis (reqts 1133640, 1133641,
        # and 1133642)
        log.info('Stash computed products for monitoring and metrics')
        other[j]['meas_efield'] = efield
        other[j]['modul_intensity'] = np.abs(efield)**2
        other[j]['unmodul_intensity'] = other[j]['meas_intensity'] - \
                                       other[j]['modul_intensity']
        other[j]['bad_efield'] = badefield
        dhmod = other[j]['modul_intensity'][dhlist[j]]
        dhunmod = other[j]['unmodul_intensity'][dhlist[j]]
        badef = badefield[dhlist[j]] # boolean with True = bad
        other[j]['mean_modul_dh'] = np.mean(dhmod[~badef])
        other[j]['mean_unmodul_dh'] = np.mean(dhunmod[~badef])
        npixmod += dhmod[~badef].size
        modulsum += np.sum(dhmod[~badef])
        unmodulsum += np.sum(dhunmod[~badef])

        # Model e-field at this DM setting
        log.info('Model e-field at this setting')
        edm0 = cfg.sl_list[j].eprop(dmlistmeas)
        ely = cfg.sl_list[j].proptolyot(edm0)
        edh0 = cfg.sl_list[j].proptodh(ely)
        model_efield = insertinto(edh0, efield.shape)
        other[j]['model_efield'] = model_efield # for reqt 1133640

        log.info('Compute difference to use to predict next iteration')
        dest = insertinto(efield - model_efield, cfg.sl_list[j].dh.e.shape)
        destlist.append(dest)
        pass
    log.info('Number of bad e-field elements: %d out of %d',
                 np.sum(bpmeas), bpmeas.size)

    # 1133642 also requires mean mod/unmod across all filters
    other['mean_modul_dh_all'] = modulsum/npixmod
    other['mean_unmodul_dh_all'] = unmodulsum/npixmod

    # 5. Compute control strategy parameters
    log.info('5. Compute control strategy parameters')
    beta = cstrat.get_regularization(iteration, prev_c)
    log.info('beta = %g', beta)
    wdm = get_wdm(cfg, dmlistmeas, tielist)
    we0 = get_we0(cfg, cstrat, subcroplist, iteration, prev_c)
    dmmultgain = cstrat.get_dmmultgain(iteration, prev_c)
    log.info('dmmultgain = %g', dmmultgain)
    jtwj = jtwj_map.retrieve_jtwj(cstrat, iteration, prev_c)

    # 6. Compute change in DM settings from electric fields
    log.info('6. Compute change in DM settings from electric fields')
    deltadm = jac_solve(jac, emeas, beta, wdm, we0, bpmeas, jtwj,
                        hconf['howfsc']['method'])
    outdmlist = inv_to_dm(deltadm*dmmultgain, cfg, dmlistmeas)

    # 7. Compute expected mean total contrast for next iteration
    # 8. Compute absolute DM settings, camera settings, probe scale factors
    log.info('7. Compute expected mean total contrast for next iteration')
    log.info('8. Compute absolute DM settings, camera settings, probe ' +
                 'scale factors')
    log.info('Compute absolute DM settings with constraints applied')
    abs_dm1 = cfg.dmlist[0].dmvobj.constrain_dm(outdmlist[0])
    abs_dm1 = remove_subnormals(abs_dm1)
    abs_dm2 = cfg.dmlist[1].dmvobj.constrain_dm(outdmlist[1])
    abs_dm2 = remove_subnormals(abs_dm2)
    log.info('Get next contrast using absolute DM settings')
    next_c = get_next_c(cfg, [abs_dm1, abs_dm2], subcroplist, cstrat.fixedbp,
                        n2clist, destlist,
                        cleanrow=hconf['excam']['cleanrow'],
                        cleancol=hconf['excam']['cleancol'])
    log.info('Expected next contrast = %g', next_c)

    log.info('Get probe scale factors')
    probeheight = cstrat.get_probeheight(iteration, prev_c)
    log.info('probeheight = %g', probeheight)
    scale_factor_list = get_scale_factor_list(hconf['probe']['dmrel_ph_list'],
                                               probeheight)
    log.info('scale factors = [%g, %g, %g, %g, %g, %g]', scale_factor_list[0],
             scale_factor_list[1], scale_factor_list[2],
             scale_factor_list[3], scale_factor_list[4],
             scale_factor_list[5],
    )

    log.info('Compute camera settings using exposure time calculator')
    gain_list = []
    exptime_list = []
    nframes_list = []
    final_optflag = 0

    for index, sequence in enumerate(hconf['hardware']['sequence_list']):
        log.info('Sequence = %s', sequence)
        innerg = []
        innere = []
        innern = []

        # Note: implicit assumption that order of hardware sequences = order
        # of SingleLambdas.  SLs are shortest to longest wavelength, as is
        # data collection, so this is valid unless we start swapping things
        # around.  (Which would violate the ordering conventions elsewhere, so
        # low risk, but want to put in text so we don't forget.)

        # Pick the brightest pixel from the projected next dark hole
        scale = get_next_c(cfg,
                           [abs_dm1, abs_dm2],
                           subcroplist,
                           cstrat.fixedbp,
                           n2clist,
                           destlist,
                           cleanrow=hconf['excam']['cleanrow'],
                           cleancol=hconf['excam']['cleancol'],
                           method=hconf['excam']['scale_method'],
                           percentile=hconf['excam']['scale_percentile'],
                           index_list=[index]
        )

        # Pick the brightest pixel from the projected next dark hole
        scale_bright = get_next_c(cfg,
                                  [abs_dm1, abs_dm2],
                                  subcroplist,
                                  cstrat.fixedbp,
                                  n2clist,
                                  destlist,
                                  cleanrow=hconf['excam']['cleanrow'],
                                  cleancol=hconf['excam']['cleancol'],
                                  method=hconf['excam']['scale_bright_method'],
                                  percentile=\
                                    hconf['excam']['scale_bright_percentile'],
                                  index_list=[index]
        )

        # 1 unprobed first
        log.info('Unprobed camera settings from calculator')
        unprobed_snr = cstrat.get_unprobedsnr(iteration, prev_c)
        log.info('scale = %g, scale_bright = %g', scale, scale_bright)
        log.info('snr = %g', unprobed_snr)
        nframes, exptime, gain, snr_out, optflag = \
          cgi_eetc.calc_exp_time(
              sequence_name=sequence,
              snr=unprobed_snr,
              scale=scale,
              scale_bright=scale_bright,
          )
        innerg.append(gain)
        innere.append(as_f32_normal(exptime))
        innern.append(nframes)
        log.info('gain = %g, exposure time = %g, number of frames = %d',
                     gain, exptime, nframes)
        log.info('snr_out %g, optflag = %d', snr_out, optflag)
        if optflag != 0:
            final_optflag = optflag
            pass

        # nprobepair probes
        log.info('Probed camera settings from calculator')
        probed_snr = cstrat.get_probedsnr(iteration, prev_c)
        pscale = scale + probeheight # mean in dark hole, no cross term
        pscale_bright = scale_bright + probeheight + \
          2*np.sqrt(probeheight)*np.sqrt(scale_bright) # worst case, use x-term
        log.info('scale = %g, scale_bright = %g', pscale, pscale_bright)
        log.info('snr = %g', probed_snr)
        nframes, exptime, gain, snr_out, optflag = \
          cgi_eetc.calc_exp_time(
              sequence_name=sequence,
              snr=probed_snr,
              scale=pscale,
              scale_bright=pscale_bright,
          )
        log.info('gain = %g, exposure time = %g, number of frames = %d',
                     gain, exptime, nframes)
        log.info('snr_out %g, optflag = %d', snr_out, optflag)
        if optflag != 0:
            final_optflag = optflag
            pass

        for k in range(nprobepair):
            innerg.append(gain)
            innere.append(as_f32_normal(exptime))
            innern.append(nframes)
            pass
        gain_list.append(innerg)
        exptime_list.append(innere)
        nframes_list.append(innern)
        pass

    # 9. Compute expected time to complete next iteration
    log.info('9. Compute expected time to complete next iteration')
    next_time = expected_time(ndm,
                              nlam,
                              param_order_to_list(exptime_list),
                              param_order_to_list(nframes_list),
                              hconf['overhead']['overdm'],
                              hconf['overhead']['overfilt'],
                              hconf['overhead']['overboth'],
                              hconf['overhead']['overfixed'],
                              hconf['overhead']['overframe'],
                              )
    log.info('Expected time to complete next iteration = %g', next_time)

    # 10. End
    if final_optflag == 0: # first optimizer succeeded every time
        stat = status_codes['nominal']
        pass
    else: # still succeeded with 2nd optimizer; failure raises exception
        stat = status_codes['LowerThanExpectedSNR']
        pass
    log.info('Return data tuple')
    return (abs_dm1,
            abs_dm2,
            scale_factor_list,
            gain_list,
            exptime_list,
            nframes_list,
            prev_c,
            next_c,
            next_time,
            stat,
            other,
    )





if __name__ == "__main__":
    pass
