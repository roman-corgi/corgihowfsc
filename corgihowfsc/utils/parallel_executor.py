from multiprocessing import Pool

def run_parallel(func, args_list, n_jobs=1, allow_nesting=False, start_method="spawn"):
    """
    Run func over args_list in parallel using multiprocessing.Pool.
    Blocks until all jobs finish (barrier).

    Args:
        func:          top-level picklable callable
        args_list:     list of tuples, each unpacked as func(*args)
        n_jobs:        number of worker processes. Use os.sched_getaffinity(0)
                       on the cluster — never hardcode or use cpu_count()
        allow_nesting: if True, use NestablePool so workers can spawn
                       their own child processes (e.g. PROPER multirun)
        start_method:  process start method. 'spawn' is safest for
                       nested/process-heavy workloads

    Returns:
        list of results in the same order as args_list
    """
    if not args_list:
        return []

    if n_jobs == 1:
        return [func(*a) for a in args_list]

    if allow_nesting:
        with NestablePool(processes=n_jobs) as pool:
            return pool.starmap(func, args_list)

    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=n_jobs) as pool:
        return pool.starmap(func, args_list)

# Alternative implementation using joblib, but proper is using multiprocessing.Pool for parallel processing in this case -> not straightforward as it would crash ... due to nested parallelism. 

# from joblib import Parallel, delayed

# def run_parallel(fn, args_list, n_jobs=1):
#     """
#     Run fn over args_list in parallel using joblib.
#     Blocks until ALL jobs finish (barrier).
    
#     Args:
#         fn: callable — must be picklable (free function or static method)
#         args_list: list of tuples — each tuple is unpacked as fn(*args)
#         n_jobs: number of workers. 1 = serial, -1 = all cores (avoid when on cluster)
    
#     Returns:
#         list of results in same order as args_list
#     """
#     if n_jobs == 1:
#         return [fn(*a) for a in args_list]
    
#     return Parallel(n_jobs=n_jobs, backend='loky')(
#         delayed(fn)(*a) for a in args_list
#     )

