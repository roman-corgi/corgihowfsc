import time
import os
import argparse
import cProfile
import pstats
import io

import logging
log = logging.getLogger(__name__)

import numpy as np
import astropy.io.fits as pyfits

import eetc
from eetc.cgi_eetc import CGIEETC

import howfsc
from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjtwj import JTWJMap

from howfsc.model.mode import CoronagraphMode

from howfsc.util.loadyaml import loadyaml
from howfsc.util.gitl_tools import param_order_to_list

from corgihowfsc.gitl.modular_gitl import howfsc_computation
from howfsc.precomp import howfsc_precomputation
from corgihowfsc.utils.saving_output import save_outputs, save_outputs_iter
from corgihowfsc.utils.output_management import save_run_config, update_yml

from scipy import sparse

from corgihowfsc.utils.parallel_executor import run_parallel
from howfsc.control.calcjacs import get_ndhpix, calcjacs_sp
from howfsc.control.calcn2c import calcn2c

eetc_path = os.path.dirname(os.path.abspath(eetc.__file__))
howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))
defjacpath = os.path.join(os.path.dirname(howfscpath), 'jacdata')


def _collect_framelist(imager, cfg, dm1_list, dm2_list, exptime_list,
                       gain_list, nframes_list, croplist, normalization_strategy,
                       get_cgi_eetc, hconf, ndm, cstrat, fracbadpix,
                       n_jobs=1, use_mpi=False, executor=None):

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
         use_mpi,
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
        executor=executor,
    )

def _get_image_worker(imager, dm1v, dm2v, exptime, gain, nframes, crop, lind,
                      peakflux, fixedbp, fracbadpix, seed_offset, use_mpi=False):
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
    if use_mpi:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) != 'forkserver':
            mp.set_start_method('forkserver', force=True)

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


def _jac_worker(cfg, ijproc, dm0list, jacmethod, num_threads):
    """
    MPI worker: compute the partial jacobian for one chunk of actuators.

    Mirrors what calcjacs_mp's inner jac_worker does, but runs inside an
    MPI service worker process.  Returns (ijproc, partial_jac) so the
    caller can reassemble without relying on result ordering.

    Args:
        cfg:        CoronagraphMode object (pickled and sent by rank 0)
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


def mpi_precompute_jac(cfg, dmset_list, cstrat, subcroplist,
                       jacmethod, n_workers=None, num_threads=None,
                       do_n2clist=False, executor=None):
    """
    MPI-distributed drop-in replacement for howfsc_precomputation (jac + jtwj).

    Splits the actuator list across all MPI service workers exactly the same
    way calcjacs_mp does (interleaved: ijlist[ip::n_workers]), dispatches via
    MPIPoolExecutor, then assembles the full jacobian on rank 0 before
    computing JTWJMap serially.

    Use when use_mpi=True and you want the idle service workers to help with
    the jacobian instead of sitting idle while rank 0 runs the full computation
    alone.

    Args:
        cfg:         CoronagraphMode object
        dmset_list:  list of 2D DM voltage arrays (current DM setting)
        cstrat:      ControlStrategy object
        subcroplist: list of 4-tuples, one per wavelength
        jacmethod:   'fast' or 'normal'
        n_workers:   number of MPI service workers to use (= num_imager_worker)
        num_threads: int or None — forwarded to each worker to set MKL_NUM_THREADS
        do_n2clist:  if True compute n2clist; for precomp_jacs_always this is
                     always False (n2clist is loaded from file separately)

    Returns:
        (jac, jtwj_map, n2clist) matching the signature of howfsc_precomputation
    """
    # Build full actuator index list (same logic as howfsc_precomputation)
    ndmact = sum(d.registration['nact'] ** 2 for d in cfg.dmlist)
    ijlist = list(range(ndmact))
    ndhpix = get_ndhpix(cfg)[-1]

    if executor is None:
        raise ValueError(
            'mpi_precompute_jac requires a caller-owned executor; '
            'this path no longer creates ad hoc MPIPoolExecutors'
        )
    if n_workers is None:
        n_workers = executor._max_workers

    # Interleaved split: worker ip gets actuators [ip, ip+n, ip+2n, ...]
    # Mirrors calcjacs_mp's list_ijproc construction exactly.
    list_ijproc = [ijlist[ip::n_workers] for ip in range(n_workers)]

    args_list = [
        (cfg, ijproc, dmset_list, jacmethod, num_threads)
        for ijproc in list_ijproc
    ]

    results = list(executor.starmap(_jac_worker, args_list))

    # Assemble partial jacobians into the full array (same as calcjacs_mp)
    jac = np.zeros((2, ndmact, ndhpix), dtype='double')
    for ijproc, partial_jac in results:
        jac[0, ijproc, :] = partial_jac[0]
        jac[1, ijproc, :] = partial_jac[1]

    # Apply DM crosstalk correction to the full assembled jacobian.
    # Mirrors the crosstalk block in calcjacs() (howfsc/control/calcjacs.py).
    # Must be done here on the full jac — per-chunk application would be wrong
    # because crosstalk couples neighboring actuators that may be in different chunks.
    get_dmind2d = cfg.sl_list[0].get_dmind2d
    dmnjk = np.array([get_dmind2d(dm_act_ij) for dm_act_ij in ijlist])
    list_HC_sparse = []
    for idm, DM in enumerate(cfg.dmlist):
        dmnjk_idm = dmnjk[dmnjk[:, 0] == idm, :]
        if DM.dmvobj.crosstalk.HC_sparse is None:
            list_HC_sparse.append(sparse.csc_matrix(sparse.eye(dmnjk_idm.shape[0])))
        else:
            k_idm = DM.dmvobj.crosstalk.k_diag(dmnjk_idm[:, 1], dmnjk_idm[:, 2])
            list_HC_sparse.append(DM.dmvobj.crosstalk.HC_sparse[k_idm, :][:, k_idm])
    HC_ijlist = sparse.block_diag(list_HC_sparse, format='csc')
    jac_xtalk = np.zeros(jac.shape)
    jac_xtalk[0, :, :] = HC_ijlist @ jac[0, :, :]
    jac_xtalk[1, :, :] = HC_ijlist @ jac[1, :, :]
    jac = jac_xtalk

    # JTWJMap is fast and serial — always done on rank 0
    jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)

    # n2clist
    n2clist = []
    for idx in range(len(cfg.sl_list)):
        nrow = subcroplist[idx][2]
        ncol = subcroplist[idx][3]
        if do_n2clist:
            n2clist.append(calcn2c(cfg, idx, nrow, ncol, dmset_list))
        else:
            n2clist.append(np.ones((nrow, ncol)))

    return jac, jtwj_map, n2clist
