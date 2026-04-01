import numpy as np
from mpi4py import MPI
from scipy import sparse
import logging

from howfsc.control.calcjtwj import JTWJMap
from howfsc.control.calcjacs import get_ndhpix
from howfsc.control.calcn2c import calcn2c

from corgihowfsc.utils.gitl_worker import (
    init_mpi_worker_state,
    run_mpi_frame_task,
    run_mpi_jac_task,
)

log = logging.getLogger(__name__)


TASK_INIT = "INIT"
TASK_FRAME = "FRAME"
TASK_JAC_CHUNK = "JAC_CHUNK"
TASK_STOP = "STOP"

def initialize_mpi_comm(): 
    """
    --> change the name to initialize_mpi_runtime_roles()

    Enter the direct MPI runtime and assign rank roles. 

    Rank 0 returns ``MPI.COMM_WORLD`` and continues through the launcher. 
    All nonzero ranks immedidately enter the worker service loop and exit when that loop finishes. 

    This function must run before constrcuting any heavyweight runtime objects so worker ranks do not execute manager-only setup code. 

    Raises
    ------
    ValueError
        If MPI mode is started with fewer than two ranks.
    SystemExit
        On worker ranks after the worker service loop terminates.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        worker_loop(comm)
        raise SystemExit(0)
    if comm.Get_size() < 2:
        raise ValueError("MPI mode requires at least 2 ranks")
    return comm


def build_worker_init_config(cfgfile, cstratfile, hconffile, backend_type,
                              mode, corgi_overrides, args):
    """ old name: build_worker_init_payload

    Build the one-time serialisable payload to initialise MPI workers. 

    The payload contains only lightweight configuration values, such as file paths, backend and mode selections, corgisim overrides and optional stellar-properties overrides. Each worker rank uses this payload to reconstruct its own local runtime state during the initial ``INIT`` step. 

    This avoids repeatedly sending heavyweights live Python objects, such as GITLImage or loaded howfsc configuration objects, across MPI task messages. 

    Returns
    -------
    dict
        Dictionary containing all the necessary information for workers to reconstruct their local runtime state.
    """
    return {
        'cfgfile': cfgfile,
        'cstratfile': cstratfile,
        'hconffile': hconffile,
        'backend_type': backend_type,
        'mode': mode,
        'corgi_overrides': corgi_overrides,
        'stellar_vmag': getattr(args, 'stellarvmag', None),
        'stellar_type': getattr(args, 'stellartype', None),
        'stellar_vmag_target': getattr(args, 'stellarvmagtarget', None),
        'stellar_type_target': getattr(args, 'stellartypetarget', None),
    }


def init_workers(comm, worker_config):
    """
    Send the one-time INIT message to all worker ranks.

    Each worker rank receives the same ``worker_config`` payload, which it uses to reconstruct its local runtime state inside the worker loop. This step prepares workers for later FRAME adn JAC_CHUNK tasks but does not execute any tasks itself.
    
    """
    log.info("MPI manager initializing %d worker ranks", comm.Get_size() - 1)
    for rank in range(1, comm.Get_size()):
        log.info("MPI manager sending INIT to rank %d", rank)
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

    The worker waits for messages from rank 0 (main process) over ``comm``. Each message contains a ``message_type`` field that determines what the worker should do.

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
        message = comm.recv(source=0)
        message_type = message['message_type']

        if message_type == TASK_INIT:
            log.info("MPI worker rank %d received INIT", comm.Get_rank())
            worker_state = init_mpi_worker_state(message['worker_config'])
            log.info("MPI worker rank %d finished INIT", comm.Get_rank())
            continue

        if message_type == TASK_STOP:
            log.info("MPI worker rank %d received STOP", comm.Get_rank())
            break

        if worker_state is None:
            raise RuntimeError('MPI worker received a task before INIT')

        task = message['task']

        if message_type == TASK_FRAME:
            log.info(
                "MPI worker rank %d starting FRAME job_id=%s",
                comm.Get_rank(),
                task['job_id'],
            )
            result = run_mpi_frame_task(worker_state, task)
            log.info(
                "MPI worker rank %d finished FRAME job_id=%s",
                comm.Get_rank(),
                task['job_id'],
            )
        elif message_type == TASK_JAC_CHUNK:
            log.info(
                "MPI worker rank %d starting JAC_CHUNK job_id=%s",
                comm.Get_rank(),
                task['job_id'],
            )
            result = run_mpi_jac_task(worker_state, task)
            log.info(
                "MPI worker rank %d finished JAC_CHUNK job_id=%s",
                comm.Get_rank(),
                task['job_id'],
            )
        else:
            raise ValueError(f"Unknown MPI task kind: {message_type}")

        comm.send(
            {'message_type': 'RESULT', 'job_id': task['job_id'], 'result': result},
            dest=0,
        )

