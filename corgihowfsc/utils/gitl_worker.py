import time
import os

import numpy as np

from corgihowfsc.utils.parallel_executor import run_parallel
from howfsc.control.calcjacs import calcjacs_sp


def _collect_framelist(imager, cfg, dm1_list, dm2_list, exptime_list,
                       gain_list, nframes_list, croplist, normalization_strategy,
                       get_cgi_eetc, hconf, ndm, cstrat, fracbadpix,
                       n_jobs=1, use_mpi=False):

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
        use_mpi=use_mpi,
    )

def _get_image_worker(imager, dm1v, dm2v, exptime, gain, nframes, crop, lind,
                      peakflux, fixedbp, fracbadpix, seed_offset):
    """
    Worker function for parallel image collection.
    Must be a top-level function (not nested) to be picklable by joblib.

    Args:
        imager: GitlImage instance
        dm1v: ndarray, absolute voltage map for DM1
        dm2v: ndarray, absolute voltage map for DM2
        exptime: float, exposure time in seconds
        gain: float, EM gain
        nframes: int, number of frames 
        crop: 4-tuple of (lower row, lower col, nrows, ncols)
        lind: int, wavelength channel index
        peakflux: float, pre-computed peak flux for this wavelength
        fixedbp: ndarray of bool, fixed bad pixel map
        fracbadpix: float, fraction of pixels to randomly mask
        seed_offset: int, unique per frame to ensure distinct but
                     reproducible bad pixel patterns across frames

    Returns:
        ndarray: simulated detector frame with bad pixels set to NaN
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
    rng = np.random.default_rng(12345 + seed_offset)
    bpmeas = rng.random(f.shape) > (1 - fracbadpix)
    f[bpmeas] = np.nan
    return f