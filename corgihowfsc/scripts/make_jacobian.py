import os
from pathlib import Path
import multiprocessing
import time
from datetime import datetime

import numpy as np
import astropy.io.fits as fits

from howfsc.model.mode import CoronagraphMode
from howfsc.control.calcjacs import calcjacs
import howfsc.util.check as check
import corgihowfsc

VALID_JACMETHODS = ['normal', 'fast']

def main():
    base_path = Path.home()  # default root — change if saving elsewhere
    mode = 'nfov_band1' # example mode — change as needed
    dark_hole = '360deg' # example mode for the dark hole — change as needed
    jacmethod = 'normal' 

    cfgfile = get_cfgfile(mode, dark_hole)
    output = get_jacobian_output_path(base_path, mode, dark_hole)

    # num_process=0 auto-selects half the machine's CPUs (see set_num_processes)
    calculate_jacobian(cfgfile, output, jacmethod=jacmethod,
                       num_process=0, num_threads=None)

def get_jacobian_output_path(base_path, mode, dark_hole):
    """Return a timestamped output path for a new Jacobian FITS file.

    Args:
        base_path (str or Path): Root directory under which
            ``corgiloop_data/jacobians/`` will be created if it does not
            already exist.
        mode (str): Coronagraph mode.
        dark_hole (str): Dark hole configuration.

    Returns:
        str: Full file path for the new Jacobian FITS file.
    """
    jacobian_dir = Path(base_path) / 'corgiloop_data' / 'jacobians'
    jacobian_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return str(jacobian_dir / f'jacobian_{mode}_{dark_hole}_{timestamp}.fits')


def get_cfgfile(mode, dark_hole):
    """Return the path to the optical model config file for a given mode.

    Args:
        mode (str): Coronagraph mode.
        dark_hole (str): Dark hole configuration.

    Returns:
        str: Full file path to `howfsc_optical_model.yaml`.
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(corgihowfsc.__file__)),
        'model', mode, f'{mode}_{dark_hole}',
        'howfsc_optical_model.yaml',
    )

def set_num_processes(num_process):
    """Resolve the number of worker processes to use for the Jacobian calculation.

    Each worker process handles a subset of actuators independently, so more
    processes shortens wall-clock time at the cost of memory. The resolved
    value follows this precedence (highest to lowest):

      1. Explicit `num_process` argument
      2. `HOWFS_CALCJAC_NUM_PROCESS` environment variable
      3. Default of 1 (single-process, no parallelism)

    Passing `num_process=0` is a special case that automatically uses half of the
    machine's logical CPU count, which is a reasonable starting point for most
    workstations without starving other processes.

    Args:
        num_process (int or None): Requested process count, or `None` to defer to
            the environment variable / default.

    Returns:
        int: Resolved process count (always >= 1).
    """
    if num_process is None:
        # Fall back to environment variable, then hard default of 1
        num_process = int(os.environ.get('HOWFS_CALCJAC_NUM_PROCESS', 1))

    check.nonnegative_scalar_integer(num_process, 'num_process', TypeError)

    # 0 is a sentinel meaning "use half the available CPUs automatically"
    if num_process == 0:
        num_process = multiprocessing.cpu_count() // 2

    return num_process


def set_num_threads(num_process, num_threads):
    """Resolve the number of MKL threads each worker process may use.

    MKL (Math Kernel Library) is the linear algebra backend used internally by
    NumPy/SciPy. By default MKL will try to use all available cores, which
    causes severe over-subscription when multiple worker *processes* are also
    running — each process would spawn its own full thread pool and the cores
    would be fighting each other. Pinning threads to 1 per process avoids this.

    The resolved value follows this precedence (highest to lowest):

      1. Explicit ``num_threads`` argument
      2. ``HOWFS_CALCJAC_NUM_THREADS`` environment variable
      3. ``MKL_NUM_THREADS`` environment variable (already set externally)
      4. If ``num_process > 1`` and nothing else is set, default to 1
         to prevent over-subscription

    Args:
        num_process (int): Resolved process count from :func:`set_num_processes`.
            Used only to decide whether to apply the over-subscription default.
        num_threads (int or None): Requested thread count, or ``None`` to defer
            to the environment variable / default.

    Returns:
        int or None: Resolved thread count, or ``None`` if the caller should
        leave MKL's own defaults untouched.
    """
    if num_threads is None:
        # Check dedicated env var first, then fall through to over-subscription guard
        num_threads = os.environ.get('HOWFS_CALCJAC_NUM_THREADS')

        if (num_threads is None
                and os.environ.get('MKL_NUM_THREADS') is None
                and num_process > 1):
            # Multiple processes with unconstrained MKL threads would
            # over-subscribe the CPU: default to 1 thread per process
            num_threads = 1

    if num_threads is not None:
        # Environment variables arrive as strings (e.g. '4'), so cast to int
        if isinstance(num_threads, str):
            num_threads = int(num_threads)
        check.positive_scalar_integer(num_threads, 'num_threads', TypeError)

    return num_threads

def calculate_jacobian(cfgfile, output, jacmethod,
                       num_process=None, num_threads=None):
    """Calculate a Jacobian and write it to a FITS file.

    The number of actuators is derived automatically from the model — all
    actuators across every DM defined in the config file are included.

    Args:
        cfgfile (str): Path to ``howfsc_optical_model.yaml``.
        output (str): Destination path for the output FITS file.
        jacmethod (str): Calculation method — 'normal' or 'fast'.
        num_process (int or None): Worker processes (see
            :func:`set_num_processes` for rules).
        num_threads (int or None): MKL threads per process (see
            :func:`set_num_threads` for rules).
    """
    check.string(jacmethod, 'jacmethod', TypeError)
    if jacmethod not in VALID_JACMETHODS:
        raise ValueError(
            f"Invalid jacmethod '{jacmethod}'. Valid options: {VALID_JACMETHODS}"
        )

    cfg = CoronagraphMode(cfgfile)

    # Derive the total actuator count by summing nact^2 across all DMs.
    # nact is the number of actuators along one axis of a square DM grid,
    # so the total per DM is nact^2, and we sum over all DMs in cfg.dmlist.
    nact = int(np.cumsum(
        [0] + [d.registration['nact'] ** 2 for d in cfg.dmlist]
    )[-1])
    ijlist = range(nact)  # one entry per actuator, iterated by calcjacs

    # dm0list=None tells calcjacs to use the initial DM map built into the cfg
    # rather than a custom starting state
    dm0list = None

    # Resolve parallelism settings from arguments / environment / defaults
    num_process = set_num_processes(num_process)
    num_threads = set_num_threads(num_process, num_threads)

    # Pin MKL thread count if needed, saving any pre-existing value so we can
    # restore it exactly after the calculation (important when this function is
    # called from within a larger script that may have its own MKL settings)
    saved_mkl = None
    if num_threads is not None:
        saved_mkl = os.environ.get('MKL_NUM_THREADS')
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        print(f'Set MKL_NUM_THREADS = {num_threads}')
    elif 'MKL_NUM_THREADS' in os.environ:
        # Already set externally. Leave it alone and just report it
        print(f"MKL_NUM_THREADS = {os.environ['MKL_NUM_THREADS']} (from environment)")

    print(f'Beginning Jacobian calculation: {num_process} process(es), method={jacmethod}')
    print(f'Total actuators: {nact}')
    t0 = time.time()

    try:
        jac = calcjacs(cfg, ijlist, dm0list, jacmethod=jacmethod,
                       num_process=num_process)
    finally:
        # Always restore MKL_NUM_THREADS even if calcjacs raises, so the
        # calling environment is not left in an unexpected state
        if num_threads is not None:
            if saved_mkl is not None:
                os.environ['MKL_NUM_THREADS'] = saved_mkl
            else:
                del os.environ['MKL_NUM_THREADS']
            print('Restored MKL_NUM_THREADS to previous value')

    print(f'Jacobian calculation complete: {time.time() - t0:.2f} seconds')

    if output is not None:
        print(f'Writing Jacobian to: {output}')
        fits.writeto(output, jac)

if __name__ == '__main__':
    main()