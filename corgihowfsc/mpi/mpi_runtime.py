import numpy as np
from mpi4py import MPI
from scipy import sparse
import logging
import os

from howfsc.control.calcjtwj import JTWJMap
from howfsc.control.calcjacs import get_ndhpix
from howfsc.control.calcn2c import calcn2c

from corgihowfsc.mpi.mpi_worker import (
    initialize_mpi_worker_state,
    run_mpi_frame_task,
    run_mpi_jac_task,
)

log = logging.getLogger(__name__)


TASK_INIT = "INIT"
TASK_FRAME = "FRAME"
TASK_JAC_CHUNK = "JAC_CHUNK"
TASK_STOP = "STOP"


def _configure_worker_logging(worker_config, rank):
    """
    Configure a rank-specific logfile for an MPI worker rank.

    Rank 0 already configures the main ``gitl.log`` through the normal launcher
    path. Worker ranks branch into the MPI worker loop before they ever reach
    that logger setup, so they need their own explicit ``basicConfig`` call.
    """
    if not worker_config.get('debug', False):
        return

    logfile = worker_config.get('logfile')
    if not logfile:
        return

    log_dir = os.path.dirname(logfile)
    os.makedirs(log_dir, exist_ok=True)
    worker_logfile = os.path.join(log_dir, f'gitl_rank{rank}.log')
    setup_logging(
        debug=worker_config.get('debug', False),
        logfile=worker_logfile,
    )
    log.info(
        "MPI worker logging configured: rank=%d pid=%d logfile=%s",
        rank,
        os.getpid(),
        worker_logfile,
    )

def initialize_mpi_comm(): 
    """
    Enter the direct MPI runtime and assign rank roles. 

    Rank 0 returns ``MPI.COMM_WORLD`` and continues through the launcher. 
    All nonzero ranks immedidately enter the worker service loop and exit when that loop finishes. 

    This function must run before constrcuting any heavyweight runtime objects so worker ranks do not execute manager-only setup code. 

    Raises: 
        ValueError: If MPI mode is started with fewer than two ranks.
        SystemExit: On worker ranks after the worker service loop terminates.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        worker_loop(comm)
        raise SystemExit(0)
    if comm.Get_size() < 2:
        raise ValueError("MPI mode requires at least 2 ranks")
    return comm


def build_worker_init_config(args, cfgfile, cstratfile, hconffile, backend_type, mode, corgi_overrides):
    """
    Build the one-time serialisable payload to initialise MPI workers. 

    The payload contains only lightweight configuration values, such as file paths, backend and mode selections, corgisim overrides and optional stellar-properties overrides. Each worker rank uses this payload to reconstruct its own local runtime state during the initial ``INIT`` step. 

    This avoids repeatedly sending heavyweights live Python objects, such as GITLImage or loaded howfsc configuration objects, across MPI task messages. 

    Returns:
        dict: Dictionary containing all the necessary information for workers to reconstruct their local runtime state.
    """
    return {
        'cfgfile': cfgfile,
        'cstratfile': cstratfile,
        'hconffile': hconffile,
        'backend_type': backend_type,
        'mode': mode,
        'corgi_overrides': corgi_overrides,
        'logfile': getattr(args, 'logfile', None),
        'stellar_vmag': getattr(args, 'stellarvmag', None),
        'stellar_type': getattr(args, 'stellartype', None),
        'stellar_vmag_target': getattr(args, 'stellarvmagtarget', None),
        'stellar_type_target': getattr(args, 'stellartypetarget', None),
    }


def initialize_workers(comm, worker_config):
    """
    Send the one-time INIT message to all worker ranks.

    Each worker rank receives the same ``worker_config`` payload, which it uses to reconstruct its local runtime state inside the worker loop. This step prepares workers for later FRAME adn JAC_CHUNK tasks but does not execute any tasks itself.
    
    """
    print("MPI manager initializing %d worker ranks", comm.Get_size() - 1)
    for rank in range(1, comm.Get_size()):
        print("MPI manager sending INIT to rank %d", rank)
        comm.send({'message_type': TASK_INIT, 'worker_config': worker_config}, dest=rank)


def shutdown_workers(comm):
    """
    Send the onet-time STOP message to all worker ranks. 

    This tells each worker to leave its persistent receive loop and exit cleanly.
    It should be called once at the end of the MPI run after all FRAME and JAC_CHUNK tasks have completed.
    
    """
    if comm is None:
        return
    log.info("MPI manager sending STOP to %d worker ranks", comm.Get_size() - 1)
    for rank in range(1, comm.Get_size()):
        comm.send({'message_type': TASK_STOP}, dest=rank)


def worker_loop(comm):
    """
    Run the persistent MPI worker loop for nonzero ranks. 

    The worker waits for messages from rank 0 (main process) over ``comm``. 
    Each message contains a ``message_type`` field that determines what the worker should do.

    The worker handles four message types:
    - ``INIT``: build local worker state once from the provided configuration
    - ``FRAME``: execute one frame-generation task and send the result back
    - ``JAC_CHUNK``: execute one Jacobian chunk task and send the result back
    - ``STOP``: exit the loop and terminate cleanly

    This keeps the worker side simple: one incoming command produces one
    outgoing result, until rank 0 sends ``STOP``.
    """
    worker_state = None

    while True:
        message = comm.recv(source=0) # wait for message from rank 0 (main process)
        message_type = message['message_type']

        if message_type == TASK_INIT:
            _configure_worker_logging(message['worker_config'], comm.Get_rank())
            log.info("MPI worker rank %d pid=%d received INIT", comm.Get_rank(), os.getpid())
            worker_state = initialize_mpi_worker_state(message['worker_config'])
            log.info("MPI worker rank %d pid=%d finished INIT", comm.Get_rank(), os.getpid())
            continue

        if message_type == TASK_STOP:
            log.info("MPI worker rank %d pid=%d received STOP", comm.Get_rank(), os.getpid())
            break

        if worker_state is None:
            raise RuntimeError('MPI worker received a task before INIT')

        task = message['task']

        if message_type == TASK_FRAME:
            log.info(
                "MPI worker rank %d pid=%d starting FRAME job_id=%s",
                comm.Get_rank(),
                os.getpid(),
                task['job_id'],
            )
            result = run_mpi_frame_task(worker_state, task)
            log.info(
                "MPI worker rank %d pid=%d finished FRAME job_id=%s",
                comm.Get_rank(),
                os.getpid(),
                task['job_id'],
            )
        elif message_type == TASK_JAC_CHUNK:
            log.info(
                "MPI worker rank %d pid=%d starting JAC_CHUNK job_id=%s",
                comm.Get_rank(),
                os.getpid(),
                task['job_id'],
            )
            result = run_mpi_jac_task(worker_state, task)
            log.info(
                "MPI worker rank %d pid=%d finished JAC_CHUNK job_id=%s",
                comm.Get_rank(),
                os.getpid(),
                task['job_id'],
            )
        else:
            raise ValueError(f"Unknown MPI task kind: {message_type}")

        comm.send(
            {'message_type': 'RESULT', 'job_id': task['job_id'], 'result': result},
            dest=0,
        ) # send result back to rank 0 (main process)


def collect_framelist_mpi(comm, imager, cfg, dm1_list, dm2_list, exptime_list,
                          gain_list, nframes_list, croplist,
                          normalization_strategy, get_cgi_eetc, hconf, ndm,
                          cstrat, fracbadpix, iteration=0, max_workers=None):
    """
    Collect detector frames for the full framelist by distributing explicit frame tasks over MPI workers.

    This is the manager-side entry point for framelist generation. Rank 0 computes any shared per-wavelength values 
    (e.g. peak flux) once, builds one ``FRAME`` task per requested output frame, then sends those tasks through the shared MPI task queue, and gather results back in logical frame order. 

    The workers do not receive heavyweight live objects with each task. They reuse the cached worker state built earlier during ``INIT`` and combine that state with the per-frame task inputs.

    Notes
    -----
    The signature intentionally looks similar to the old local framelist path
    so that the call sites in ``nulling_gitl.py`` can stay simple. Some inputs
    such as ``imager`` and ``cstrat`` are present for interface consistency
    even though the MPI manager itself does not use them directly here.

    ``max_workers`` caps the number of active MPI workers used for this queue.
    In practice this is how ``num_imager_worker`` is enforced when the job is
    launched with more MPI ranks than the application wants to use.

    Args: 
        max_workers : int or None, optional
            Maximum number of MPI worker ranks to use for this queue. This caps the
            active worker count even if the MPI job was launched with more ranks.

    Returns: 
        list: Ordered list of generated detector frames, one element per requested frame task.
    """
    if comm is None:
        raise ValueError('collect_framelist_mpi requires an MPI communicator')

    # Peak flux depends only on wavelength here, so it is cheaper to compute
    # these values once on rank 0 rather than repeating them inside every task.
    peakflux_list = [
        normalization_strategy.calc_flux_rate(
            get_cgi_eetc, hconf, indj, dm1_list[0], dm2_list[0], gain=1
        )[1]
        for indj in range(len(cfg.sl_list))
    ]

    # Build one FRAME task per output frame --> same as the _collect_framelist
    frame_tasks = [
        {
            'job_id': indj * ndm + indk,
            'dm1v': dm1_list[indj * ndm + indk],
            'dm2v': dm2_list[indj * ndm + indk],
            'exptime': exptime_list[indj * ndm + indk],
            'gain': gain_list[indj * ndm + indk],
            'nframes': nframes_list[indj * ndm + indk],
            'crop': croplist[indj],
            'lind': indj,
            'peakflux': peakflux_list[indj],
            'fracbadpix': fracbadpix,
            'iteration': iteration,
            'seed_offset': indj * ndm + indk,
        }
        for indj in range(len(cfg.sl_list))
        for indk in range(ndm)
    ]
    log.info("MPI manager starting FRAME queue with %d jobs", len(frame_tasks))

    return _run_manager_task_queue(comm, TASK_FRAME, frame_tasks, max_workers=max_workers)


def precompute_jac_mpi(comm, cfg, dmset_list, cstrat, subcroplist,
                       jacmethod, num_threads=None, do_n2clist=False,
                       max_workers=None):
    """
    Precompute the Jacobian by distributing explicit JAC_CHUNK tasks over MPI workers, 
    using the exact same setup from cgihowfsc to setup the jacobian computation. 

    This is the manager-side entry point for Jacobian precomputation. Rank 0 splits the full actuator index list into chunks, sends one explicit ``JAC_CHUNK`` task per chunk through the shared MPI task queue, and gathers the partial Jacobians back to reassemble the full Jacobian on rank 0 (main process).

    After gathering the worker results, rank 0 applies the remaining steps locally. 

    ``max_workers`` is the manager-side cap on how many MPI worker ranks are
    used. The actual active worker count is:

    ``min(comm.Get_size() - 1, max_workers, ndmact)``

    where ``ndmact`` is the number of actuator indices to process.

    Args: 
        max_workers : int or None, optional
            Maximum number of MPI worker ranks to use. The active worker count is
            limited by both this value and the number of actuator chunks.

    Returns:
        tuple: ``(jac, jtwj_map, n2clist)`` where ``jac`` is the assembled Jacobian, 
            ``jtwj_map`` is the rank-0 ``JTWJMap`` object, and ``n2clist`` is either the computed list or ``None`` if ``do_n2clist`` is false.


    """
    if comm is None:
        raise ValueError('precompute_jac_mpi requires an MPI communicator')

    ndmact = sum(d.registration['nact'] ** 2 for d in cfg.dmlist)
    ijlist = list(range(ndmact))
    ndhpix = get_ndhpix(cfg)[-1]

    # MPI launch size is the upper bound, but the application can request a
    # smaller active worker count via num_imager_worker / max_workers.
    available_workers = comm.Get_size() - 1
    if max_workers is not None:
        available_workers = min(available_workers, max_workers)
    n_workers = min(available_workers, max(1, ndmact))

    # Interleave actuator indices across workers instead of assigning one large
    # contiguous block to each worker. This mirrors the old Jacobian split
    # strategy and helps spread actuator work more evenly.
    list_ijproc = [ijlist[ip::n_workers] for ip in range(n_workers)]

    tasks = [
        {
            'job_id': job_id,
            'ijproc': ijproc,
            'dmset_list': dmset_list,
            'jacmethod': jacmethod,
            'num_threads': num_threads,
        }
        for job_id, ijproc in enumerate(list_ijproc)
        if ijproc
    ]

    log.info("MPI manager starting JAC_CHUNK queue with %d jobs", len(tasks))
    results = _run_manager_task_queue(comm, TASK_JAC_CHUNK, tasks, max_workers=n_workers)

    # Below is the implementation from cgihowfsc to reassemble the Jacobian and apply the remaining steps on rank 0 after gathering the worker results.
    # Reassemble the full Jacobian on rank 0. Each worker returns exactly one
    # ``(ijproc, partial_jac)`` pair, so result placement is explicit.
    jac = np.zeros((2, ndmact, ndhpix), dtype='double')
    for ijproc, partial_jac in results:
        jac[0, ijproc, :] = partial_jac[0]
        jac[1, ijproc, :] = partial_jac[1]

    # Crosstalk must be applied after assembly because neighboring actuators
    # can belong to different worker chunks.
    get_dmind2d = cfg.sl_list[0].get_dmind2d
    dmnjk = np.array([get_dmind2d(dm_act_ij) for dm_act_ij in ijlist])
    list_hc_sparse = []
    
    for idm, dm in enumerate(cfg.dmlist):
        dmnjk_idm = dmnjk[dmnjk[:, 0] == idm, :]
        if dm.dmvobj.crosstalk.HC_sparse is None:
            list_hc_sparse.append(sparse.csc_matrix(sparse.eye(dmnjk_idm.shape[0])))
        else:
            k_idm = dm.dmvobj.crosstalk.k_diag(dmnjk_idm[:, 1], dmnjk_idm[:, 2])
            list_hc_sparse.append(dm.dmvobj.crosstalk.HC_sparse[k_idm, :][:, k_idm])
    
    hc_ijlist = sparse.block_diag(list_hc_sparse, format='csc')
    jac_xtalk = np.zeros(jac.shape)
    jac_xtalk[0, :, :] = hc_ijlist @ jac[0, :, :]
    jac_xtalk[1, :, :] = hc_ijlist @ jac[1, :, :]
    jac = jac_xtalk

    # JTWJMap remains serial on rank 0.
    jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)

    n2clist = []
    for idx in range(len(cfg.sl_list)):
        nrow = subcroplist[idx][2]
        ncol = subcroplist[idx][3]
        if do_n2clist:
            n2clist.append(calcn2c(cfg, idx, nrow, ncol, dmset_list))
        else:
            n2clist.append(np.ones((nrow, ncol)))

    return jac, jtwj_map, n2clist


def _run_manager_task_queue(comm, task_type, task_list, max_workers=None):
    """
    Run a manager-side MPI task queue and return results in logical task order.

    Rank 0 uses it to:
    - send one initial task to each active worker rank
    - wait for whichever worker finishes first
    - immediately send the next pending task to that now-free worker
    - continue until all tasks have completed
    - return results in job_id order, even if workers finish out of order

    Args: 
        comm: The MPI communicator, typically ``MPI.COMM_WORLD``.
        message_type: The task type to send to workers, for example ``FRAME`` or ``JAC_CHUNK``.
        task_list: A list of explicit task dictionaries. Each task must contain a ``job_id`` field.
        max_workers: Optional cap on how many worker ranks rank 0 should actively use.

    Returns:
        list: Task results ordered by ``job_id``.

    Raises:
        ValueError: If no worker ranks are available.
    """
    if not task_list:
        return []

    active_worker_count = comm.Get_size() - 1
    
    if max_workers is not None:
        active_worker_count = min(active_worker_count, max_workers)
    
    # ``ordered_results`` is indexed by job_id, so returned results are placed in their
    # original logical order even though workers may finish out of order.
    ordered_results = [None] * len(task_list)

    next_task_index = 0

    # Fill the worker pool: send one initial task to each active worker rank.
    # After this point, all active workers should be busy.
    for rank in range(1, active_worker_count + 1):
        if next_task_index >= len(task_list):
            break
        
        task = task_list[next_task_index]
        log.info(
            "MPI manager dispatching %s job_id=%s to rank %d",
            task_type,
            task['job_id'],
            rank,
        )
        comm.send({'message_type': task_type, 'task': task}, dest=rank)
        next_task_index += 1

    completed_task_count = 0
    status = MPI.Status()

    # Get results and keep refilling workers with pending tasks as they finish.
    # This is the steady-state queueing phase.
    while completed_task_count < len(task_list):
        message = comm.recv(source=MPI.ANY_SOURCE, status=status)
        rank = status.Get_source()
        log.info(
            "MPI manager received %s result for job_id=%s from rank %d",
            task_type,
            message['job_id'],
            rank,
        )
        ordered_results[message['job_id']] = message['result']
        completed_task_count += 1
        
        if next_task_index < len(task_list):
            # The worker that just returned a result is now free, so immediately give it
            # the next pending task.
            task = task_list[next_task_index]
            log.info(
                "MPI manager dispatching %s job_id=%s to rank %d",
                task_type,
                task['job_id'],
                rank,
            )
            comm.send({'message_type': task_type, 'task': task}, dest=rank)
            next_task_index += 1

    return ordered_results
