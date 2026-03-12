from multiprocessing import Pool

def run_parallel(fn, args_list, n_jobs=1):
    """
    Run fn over args_list in parallel using multiprocessing.Pool.
    Blocks until ALL jobs finish (barrier).

    Args:
        fn: callable — must be top-level (picklable)
        args_list: list of tuples — each tuple unpacked as fn(*args)
        n_jobs: number of workers. Set from os.sched_getaffinity(0).
                Never hardcode or use -1.

    Returns:
        list of results in same order as args_list
    """
    if n_jobs == 1:
        return [fn(*a) for a in args_list]

    with Pool(processes=n_jobs) as pool:
        return pool.starmap(fn, args_list)

# Alternative implementation using joblib, but proper is using multiprocessing.Pool for parallel processing in this case. 

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

