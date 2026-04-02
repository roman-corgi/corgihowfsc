import time
import os

import numpy as np

from corgihowfsc.utils.parallel_executor import run_parallel
from howfsc.control.calcjacs import calcjacs_sp


def _collect_framelist(imager, cfg, dm1_list, dm2_list, exptime_list,
                       gain_list, nframes_list, croplist, normalization_strategy,
                       get_cgi_eetc, hconf, ndm, cstrat, fracbadpix,
                       n_jobs=1):
    """
    Generate and collect the full framelist using local multiprocessing. 
    This helper is the local, non MPI version of framelist collection.

    Compute the peakflux per wavelength channel, build a flat list of arguments for each requested frame, and then
    execute the frame generation in parallel using ``run_parallel(...)`` with the low-level worker function
    ``_get_image_worker(...)``.

    Args: 
        imager : GitlImage
            Image generation object used to simulate detector frames.
        cfg : CoronagraphMode
            Optical model configuration. Only ``cfg.sl_list`` is used here to infer
            the number of wavelength channels.
        dm1_list, dm2_list : list
            Lists of absolute DM1 and DM2 settings for every frame in the probing
            sequence, ordered consistently with the requested framelist.
        exptime_list, gain_list, nframes_list : list
            Per frame camera settings passed through to the image worker.
        croplist : list
            Per wavelength crop definitions. The element at index ``indj`` is reused
            for all DM settings at that wavelength.
        normalization_strategy : object
            Normalisation helper used to compute the peak flux per wavelength.
        get_cgi_eetc : object
            Exposure time calculator instance used by
            ``normalization_strategy.calc_flux_rate(...)``.
        hconf : dict
            Hardware and stellar configuration dictionary passed to the
            normalisation logic.
        ndm : int
            Number of DM settings per wavelength in the probing sequence.
        cstrat : ControlStrategy
            Control strategy object. Only ``cstrat.fixedbp`` is used here.
        fracbadpix : float
            Fraction of additional random bad pixels to inject into each generated
            frame.
        n_jobs : int, optional
            Number of local worker processes to use. ``n_jobs=1`` runs serially.

    Returns: 
        list: Ordered list of simulated detector frames, one element per requested frame in the probing sequence.
    """

    # pre-compute peakflux per wavelength before parallelising
    peakflux_list = [
        normalization_strategy.calc_flux_rate(
            get_cgi_eetc, hconf, indj, dm1_list[0], dm2_list[0], gain=1
        )[1]
        for indj in range(len(cfg.sl_list))
    ]

    # flat args list — one tuple per frame
    # indj * ndm + indk -> frame_index

    args_list = [
        (imager,
         dm1_list[indj * ndm + indk],
         dm2_list[indj * ndm + indk],
         exptime_list[indj * ndm + indk],
         gain_list[indj * ndm + indk],
         nframes_list[indj * ndm + indk],
         croplist[indj],
         indj,
         peakflux_list[indj],
         cstrat.fixedbp,
         fracbadpix,
         indj * ndm + indk,   # seed_offset
        )
        for indj in range(len(cfg.sl_list))
        for indk in range(ndm)
    ]

    return run_parallel(
        _get_image_worker,
        args_list,
        n_jobs=n_jobs,
        allow_nesting=True,
        start_method="spawn",
    )

def _get_image_worker(imager, dm1v, dm2v, exptime, gain, nframes, crop, lind,
                      peakflux, fixedbp, fracbadpix, seed_offset):
    """
    Generate a frame for a single wavelength and DM setting.

    This is the low-level frame generation function shared by the local multiprocessing and MPI runtime. 

    The worker performs three steps: it calls ``imager.get_image(...)`` to
    generate the detector frame, builds a reproducible random bad pixel mask
    using ``seed_offset``, and replaces the selected pixels with ``NaN``.

    Args:
        imager : GitlImage
            Image generation object used to simulate the detector frame.
        dm1v, dm2v : ndarray
            Absolute voltage maps for DM1 and DM2 for this frame.
        exptime : float
            Exposure time in seconds.
        gain : float
            EM gain used for this frame.
        nframes : int
            Number of detector frames to average in the image simulation.
        crop : tuple
            Four element crop definition ``(lower_row, lower_col, nrows, ncols)``.
        lind : int
            Wavelength channel index.
        peakflux : float
            Precomputed peak flux for this wavelength channel.
        fixedbp : ndarray of bool
            Fixed bad pixel map passed to the image generator.
        fracbadpix : float
            Fraction of additional random bad pixels to inject into the generated
            frame.
        seed_offset : int
            Per frame seed offset used to make the random bad pixel pattern distinct
            but reproducible across frames.

    Returns:
        ndarray: Simulated detector frame with injected bad pixels set to ``NaN``.
    """
    f = imager.get_image(
        dm1v, dm2v, exptime,
        gain=gain,
        nframes=nframes,
        crop=crop,
        lind=lind,
        peakflux=peakflux,
        cleanrow=1024,
        cleancol=1024,
        fixedbp=fixedbp,
        wfe=None,
    )

    # Build a reproducible per frame random bad pixel mask.
    rng = np.random.default_rng(12345 + seed_offset)
    bpmeas = rng.random(f.shape) > (1 - fracbadpix)
    f[bpmeas] = np.nan

    return f


def _jac_worker(cfg, ijproc, dm0list, jacmethod, num_threads):
    """
    Compute the partial Jacobian for one chunk of actuators.

    This is the low-level Jacobian kernel shared by MPI-facing helpers. It
    mirrors the chunk-level logic used by the HOWFSC Jacobian multiprocessing
    path but returns ``(ijproc, partial_jac)`` explicitly so the caller can
    reassemble results without relying on completion order.

    Args:
        cfg:        CoronagraphMode object already available to the caller
        ijproc:     list of actuator indices assigned to this worker
        dm0list:    list of 2D DM voltage arrays (current DM setting)
        jacmethod:  'fast' or 'normal'
        num_threads: int or None — sets MKL_NUM_THREADS on the worker to
                     prevent MKL from spawning excess threads inside each
                     MPI worker (mirrors howfsc_precomputation's handling)

    Returns:
        (ijproc, partial_jac) where partial_jac has shape (2, len(ijproc), ndhpix)
    """
    if num_threads is not None:
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
    return ijproc, calcjacs_sp(cfg, ijproc, dm0list, jacmethod)

