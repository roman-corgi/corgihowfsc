# MPI and Multiprocessing

This page explains how parallelism works in `corgihowfsc`, when to use each mode, and how to size a run correctly.

For the YAML field descriptions, see [Parameter Reference](parameter_reference.md).

---

## What is MPI and why do we need it?

**MPI (Message Passing Interface)** is a standard communication protocol for parallel programs. Unlike Python's built-in `multiprocessing`, which is limited to one machine, MPI can coordinate processes across many nodes in a cluster, each with its own memory, by passing messages over a high-speed network.

`corgihowfsc` needs MPI because a single simulation iteration (one set of probe frames + a Jacobian update) can be computationally expensive. On a cluster with many nodes, MPI lets you distribute the work across them so that more frames and Jacobian chunks are computed simultaneously, reducing wall-clock time.

In practice:

- `multiprocessing` is simpler and sufficient for development, debugging, or single-machine runs
- MPI is the right tool when a single node is a bottleneck and you have access to a multi-node allocation (e.g. via SLURM)

---

## Overview

`corgihowfsc` has two parallel execution modes. They differ in **where the workers live** and **how work is dispatched** to them:

| Mode | When to use | How to launch |
| --- | --- | --- |
| **Local multiprocessing** | Single machine, laptop, debugging | `python scripts/run_corgisim_nulling_gitl.py` |
| **MPI** | Multi-node cluster, or when you want strict rank-level control | `mpiexec -n N python scripts/run_corgisim_nulling_gitl.py` |

The switch between them is a single YAML field:

```yaml
runtime:
  use_mpi: false   # local multiprocessing
  use_mpi: true    # MPI
  debug: false     # optional verbose logging and extra debug outputs
```

Both modes can run the same loop. MPI is not required for correctness, only for scaling beyond one machine or for workloads that saturate a single node.
When `debug: true`, the run writes more verbose logs and may create extra debug artifacts. In MPI mode, this includes rank-specific worker log files.

---

## Local multiprocessing

### How it works

When `use_mpi: false`, all parallelism is handled by `multiprocessing.Pool` inside a single process. Two independent axes of parallelism are in play:

**Outer axis — imager workers** (`num_imager_worker`)  
Multiple probe images can be collected in parallel. Each worker is an independent OS process that receives one (DM setting, wavelength) task and returns a detector frame.

**Inner axis — PROPER processes** (`num_proper_process`)  
Each imager worker may itself spawn a `multiprocessing.Pool` internally for PROPER's optical propagation. This is nested parallelism: an outer worker spawning its own inner workers.

To allow this nesting, `corgihowfsc` uses `NestablePool` (see `utils/parallel_executor.py`), which overrides the default behaviour that prevents daemonic processes from spawning children.

**Jacobian parallelism** (`num_jac_process`)  
Jacobian computation uses a separate pool that is independent from the imager workers. It only applies in local mode, in MPI mode the Jacobian is distributed across ranks instead.

### CPU sizing

Peak concurrent CPU usage on one machine is:

```text
num_imager_worker × num_proper_process
```

At startup, `get_cpu_allocation()` checks this product against the CPUs allocated to the process and warns if it exceeds the hardware limit. On Linux clusters it reads `sched_getaffinity`; on macOS/Windows it falls back to `cpu_count()`.

**Example**: 3 imager workers × 5 PROPER processes = 15 CPUs peak. If your node or session has 16 CPUs, this is fine. If you have 8, expect hardware thrashing.

---

## MPI

### MPI execution model

When `use_mpi: true`, the application uses a **manager–worker** model across MPI ranks:

- **Rank 0** runs the main loop (`nulling_gitl`) and acts as the task manager
- **Ranks 1..N−1** are persistent workers that wait for tasks from rank 0

All communication goes through `mpi4py` point-to-point `send`/`recv` calls. There is no collective communication during the loop.

### Worker lifecycle

Workers are long-lived: they are initialized once and then reused for all frames and Jacobian chunks in the run. The lifecycle has four stages:

```text
rank 0                           rank 1..N-1
------                           -----------
initialize_mpi_comm()            initialize_mpi_comm()
  (rank != 0 check passes,         (rank != 0 check fails,
   returns comm)                    enters worker_loop(), blocks on recv())

send INIT + worker_config    →   recv INIT, build local GitlImage,
                                 cfg, cstrat, hconf
                                 (state is cached for the rest of the run)

for each probe frame:
  send FRAME task              →   run frame, send RESULT back
  ← recv RESULT

for each Jacobian chunk:
  send JAC_CHUNK task          →   run Jacobian slice, send RESULT back
  ← recv RESULT

send STOP                      →   recv STOP, exit worker_loop(), terminate
```

The `worker_config` data sent during `INIT` contains only lightweight values (file paths, backend type, mode, stellar overrides), not live Python objects. Each worker rebuilds its own `GitlImage` and configuration objects locally. This avoids serialising and transmitting large objects with every task.

### Task queue

Rank 0 uses a dynamic task queue (`_run_manager_task_queue`) to keep workers busy:

1. One initial task is sent to each active worker rank
2. Rank 0 waits on `recv(source=ANY_SOURCE)`
3. Whichever worker finishes first sends its result back
4. Rank 0 immediately dispatches the next pending task to that now-free worker
5. Results are reordered by `job_id` before being returned, so the caller always gets frames back in logical order regardless of which worker finished first

This keeps all workers busy: as soon as one finishes, it gets the next task immediately.

### Rank sizing

As a rule of thumb:

```text
number of MPI ranks = num_imager_worker + 1
```

The `+1` is for rank 0, which does not execute frame or Jacobian tasks. `num_imager_worker` caps how many worker ranks are actively used even if the job is launched with more ranks than that.

**Example**: `num_imager_worker: 21` → launch with `-n 22`.

If you launch with more ranks than `num_imager_worker + 1`, the extra ranks are still initialised but will not receive tasks. This wastes allocation.

---

## Choosing a mode

| Situation | Recommendation |
| --- | --- |
| Laptop or single workstation, any model | `use_mpi: false` |
| Debugging, first run, troubleshooting | `use_mpi: false` |
| `cgi-howfsc` (compact) model on a cluster | Either mode works; local multiprocessing is usually sufficient |
| `corgihowfsc` (corgisim) model, many iterations | `use_mpi: true` — each frame is slow enough to justify distributed workers |
| Multi-node job scheduler (SLURM) | `use_mpi: true` with `mpiexec` |

Start with `use_mpi: false`. Switch to MPI when the per-iteration wall time is the bottleneck and you have multiple nodes available.

---

## Code layout

| File | Role |
| --- | --- |
| `corgihowfsc/scripts/run_corgisim_nulling_gitl.py` | Launcher — reads YAML, calls `initialize_mpi_comm()` before any setup |
| `corgihowfsc/gitl/nulling_gitl.py` | Main loop — calls either the local or MPI frame/Jacobian path depending on `comm` |
| `corgihowfsc/mpi/mpi_runtime.py` | Manager and worker loop — `initialize_mpi_comm`, `initialize_workers`, `worker_loop`, `collect_framelist_mpi`, `precompute_jac_mpi`, `_run_manager_task_queue` |
| `corgihowfsc/mpi/mpi_worker.py` | Worker task execution — `initialize_mpi_worker_state`, `run_mpi_frame_task`, `run_mpi_jac_task` |
| `corgihowfsc/utils/gitl_worker.py` | Shared lower-level helpers used by both local and MPI paths |
| `corgihowfsc/utils/parallel_executor.py` | `NestablePool` and `run_parallel` for local multiprocessing |
| `corgihowfsc/utils/howfsc_initialization.py` | `get_cpu_allocation` — validates CPU counts and warns on oversubscription |

---

## Common issues

**The job hangs immediately at launch**  
You set `use_mpi: true` but launched without `mpiexec`, `mpirun`, or `srun`. All ranks end up as rank 0 and none enter the worker loop.

**Workers block waiting for tasks that never arrive**  
The MPI job was launched with fewer ranks than `num_imager_worker + 1`. Rank 0 tries to dispatch to ranks that do not exist.

**High CPU usage but slow wall time**  
`num_imager_worker × num_proper_process` exceeds your CPU allocation. Reduce one or both. Check the startup warning from `get_cpu_allocation`.

**Frames out of order or missing**  
Not a normal failure mode: the task queue reorders by `job_id` before returning. If you see this, check that all MPI ranks are still alive (a worker crash will cause rank 0 to hang waiting for a result that will never arrive).
