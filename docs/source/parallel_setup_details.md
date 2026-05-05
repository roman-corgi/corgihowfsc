# Parallel Runs: How Local and MPI Setup Work

This page explains how the local multiprocessing and MPI setups are arranged inside `corgihowfsc`.

For user-facing instructions on choosing local vs MPI runs and sizing jobs, see [Parallel Runs: Local and Multi-Node MPI](mpi_multiprocessing.md). For generated API references, see the module pages such as [MPI API](mpi_runtime.md), [Parallel Executor](parallel_executor.md), and [HOWFSC Initialization](howfsc_initialization.md).

:::{note}
Start with [Parallel Runs: Local and Multi-Node MPI](mpi_multiprocessing.md) if you only need to choose settings for a run. This page explains how the local and MPI setups work internally.
:::

---

## Overview

`corgihowfsc` has two parallel modes. They differ in **where the workers live** and **how work is dispatched** to them:

| Mode | Where workers live | Main implementation |
| --- | --- | --- |
| **Local multiprocessing** | On one machine, as Python child processes | `utils/parallel_executor.py`, `utils/gitl_worker.py` |
| **MPI** | Across launched MPI processes, often on multiple nodes | `mpi/mpi_runtime.py`, `mpi/mpi_worker.py` |

The switch between them is a single YAML field:

```yaml
runtime:
  use_mpi: false   # local multiprocessing
  use_mpi: true    # MPI
```

Both modes run the same GITL loop. The main difference is how frame-generation and Jacobian tasks are sent to workers.

---

## Local Multiprocessing

### Execution Model

When `use_mpi: false`, all parallelism is handled by `multiprocessing.Pool` inside a single process. Two independent axes of parallelism are in play:

**Outer axis — imager workers** (`num_imager_worker`)  
Multiple probe images can be collected in parallel. Each worker is an independent OS process that receives one (DM setting, wavelength) task and returns a detector frame.

**Inner axis — PROPER processes** (`num_proper_process`)  
Each imager worker may itself spawn a `multiprocessing.Pool` internally for PROPER's optical propagation. This is nested parallelism: an outer worker spawning its own inner workers.

To allow this nesting, `corgihowfsc` uses `NestablePool` (see `utils/parallel_executor.py`), which overrides the default behaviour that prevents daemonic processes from spawning children.

**Jacobian parallelism** (`num_jac_process`)  
Jacobian computation uses a separate pool that is independent from the imager workers. It only applies in local mode, in MPI mode the Jacobian is distributed across ranks instead (e.g. divided into 5 jobs when `num_imager_worker = 5`)

This creates nested process parallelism:

```text
local image workers x PROPER subprocesses
```

For example:

```yaml
num_imager_worker: 3
num_proper_process: 5
```

can use up to:

```text
3 x 5 = 15 CPU cores
```

### Nested Process Pools

Python's standard `multiprocessing.Pool` creates daemonic worker processes, and daemonic processes are not allowed to create their own child processes.

That is a problem for the CorgiSim backend because each imager worker may call PROPER, which internally uses a multiprocessing pool.

To support this, local frame generation uses `NestablePool` from `utils/parallel_executor.py`. `NestablePool` creates non-daemonic worker processes so those workers can spawn PROPER/CorgiSim subprocesses.

### Local Frame Tasks

Local frame generation uses `_collect_framelist()` in `utils/gitl_worker.py`.

The local path:

1. Computes shared per-wavelength values such as peak flux.
2. Builds one argument tuple per requested detector frame.
3. Runs `_get_image_worker()` over those tasks with `run_parallel()`.
4. Returns frames in logical order.

Each image task includes values such as:

- DM1 and DM2 settings
- exposure time
- gain
- number of detector frames
- crop settings
- wavelength index
- peak flux
- random seed offset

### Local Jacobian Tasks

Local Jacobian computation is controlled by `num_jac_process`.

In local mode, `num_jac_process` is passed into HOWFSC precomputation as `num_process`. This controls the local process pool used for Jacobian computation.

This process pool is separate from the local imager workers. Image generation and Jacobian computation are different phases of the loop, so their process counts should be sized independently.

### Local CPU Checks

At startup, local mode calls `get_cpu_allocation()` to validate the configured process counts and warn about likely oversubscription.

Oversubscription means asking for more runnable processes than available CPU cores. This can make a run slower instead of faster because processes compete for the same cores.

---

## MPI

### Execution Model

When `use_mpi: true`, the application uses a **manager-worker** model across MPI processes.

MPI documentation usually calls each MPI process a **rank**. In implementation terms:

- **Rank 0** runs the GITL loop (`nulling_gitl`) and acts as the manager.
- **Ranks 1..N-1** are persistent workers that wait for tasks from rank 0.

All communication goes through `mpi4py` point-to-point `send`/`recv` calls. There is no collective communication during the loop.

### Worker Lifecycle

Workers are long-lived. They are initialized once and then reused for all frames and Jacobian chunks in the run.

```text
rank 0                           rank 1..N-1
------                           -----------
initialize_mpi_comm()            initialize_mpi_comm()
  (rank 0 returns comm,            (nonzero ranks enter
   continues launcher)              worker_loop() and block on recv)

send INIT + worker_config    ->   recv INIT, build local GitlImage,
                                  cfg, cstrat, hconf
                                  (state is cached for the rest of the run)

for each probe frame:
  send FRAME task             ->   run frame, send RESULT back
  <- recv RESULT

for each Jacobian chunk:
  send JAC_CHUNK task         ->   run Jacobian slice, send RESULT back
  <- recv RESULT

send STOP                    ->    recv STOP, exit worker_loop(), terminate
```

### Worker Configuration

The `worker_config` data sent during `INIT` contains only lightweight values:

- file paths
- backend type
- mode
- corgisim overrides
- debug/logging settings
- optional stellar overrides

The manager does not send heavy live Python objects such as `GitlImage`, `CoronagraphMode`, or loaded HOWFSC configuration objects with each task. Instead, each worker rebuilds those objects locally during `INIT`, caches them, and reuses them for later tasks.

### MPI Message Types

The MPI setup uses four manager-to-worker message types:

| Message | Purpose |
| --- | --- |
| `INIT` | Build worker-local state once at startup |
| `FRAME` | Generate one detector frame task |
| `JAC_CHUNK` | Compute one chunk of the Jacobian |
| `STOP` | Tell the worker to exit cleanly |

Workers send results back to the manager with the original `job_id`, which lets the manager restore logical output order.

### Manager Task Scheduling

Rank 0 uses `_run_manager_task_queue()` as a manager-side scheduling loop to keep workers busy:

1. One initial task is sent to each active worker rank.
2. Rank 0 waits on `recv(source=ANY_SOURCE)`.
3. Whichever worker finishes first sends its result back.
4. Rank 0 immediately dispatches the next pending task to that now-free worker.
5. Results are placed into the output list by `job_id` before being returned.

The frame and Jacobian task builders assign dense, zero-based `job_id` values, so this restores deterministic output order even when workers finish out of order.

### MPI Frame Tasks

MPI frame generation uses `collect_framelist_mpi()`.

The manager builds one `FRAME` task per requested detector frame and sends those tasks through the dynamic task queue. Workers use their cached `GitlImage` state plus task-specific values such as DM settings, exposure time, gain, wavelength index, and peak flux.

### MPI Jacobian Tasks

MPI Jacobian computation uses `precompute_jac_mpi()`.

The manager:

1. Builds the full actuator index list.
2. Splits actuator indices across active workers.
3. Sends one `JAC_CHUNK` task per chunk.
4. Receives partial Jacobians from workers.
5. Reassembles the full Jacobian on rank 0.
6. Applies remaining serial steps such as crosstalk handling and `JTWJMap` construction.

`num_jac_process` is local-mode only. In MPI mode, Jacobian chunks use the same active worker-rank cap as frame generation: `num_imager_worker`.

The active MPI Jacobian worker count is effectively:

```text
min(launched worker ranks, num_imager_worker, number of actuator chunks)
```

### MPI Worker Sizing Checks

MPI mode calls `validate_mpi_allocation()` after the communicator is initialized.

This check warns when:

- fewer MPI worker ranks were launched than `num_imager_worker`
- more MPI worker ranks were launched than `num_imager_worker`
- `num_proper_process` exceeds the CPUs visible to each MPI process

If too many workers are launched, extra workers are initialized but do not receive frame or Jacobian tasks. This is not a correctness issue, but it wastes allocation.

If too few workers are launched, the run continues with fewer workers and may be slower than expected.

---

## Shared Worker Functions

Both local and MPI paths reuse lower-level worker functions from `utils/gitl_worker.py`.

| Function | Used by | Purpose |
| --- | --- | --- |
| `_get_image_worker()` | Local and MPI frame paths | Generate one detector frame |
| `_jac_worker()` | MPI Jacobian path and HOWFSC-style chunking | Compute one partial Jacobian chunk |

This keeps the frame and Jacobian task behavior consistent between local and MPI modes.

---

## Code Layout

| File | Role |
| --- | --- |
| `corgihowfsc/scripts/run_corgisim_nulling_gitl.py` | Launcher; reads YAML and initializes MPI when requested |
| `corgihowfsc/gitl/nulling_gitl.py` | Main GITL loop; calls local or MPI frame/Jacobian paths |
| `corgihowfsc/utils/parallel_executor.py` | Local multiprocessing helpers, including `NestablePool` |
| `corgihowfsc/utils/gitl_worker.py` | Shared low-level frame and Jacobian worker functions |
| `corgihowfsc/utils/howfsc_initialization.py` | Local CPU allocation validation |
| `corgihowfsc/mpi/mpi_runtime.py` | MPI communicator setup, manager queue, worker loop, frame/Jacobian dispatch |
| `corgihowfsc/mpi/mpi_worker.py` | MPI worker-local initialization and task execution |

---

## Common Issues

**Local run slows down when adding workers**  
The run may be oversubscribed. Reduce `num_imager_worker`, `num_proper_process`, or `num_jac_process`.

**MPI job launches extra workers**  
The run is still correct, but extra workers will sit idle. Launch exactly `num_imager_worker + 1` total MPI ranks when possible.

**MPI job launches too few workers**  
The task queues use the workers that exist and emit a startup warning. The run is still correct, but slower than requested.

**MPI manager hangs waiting for a result**  
A worker may have crashed before returning a result. Check worker logs, scheduler stderr, memory limits, missing files, and environment consistency across nodes.

**Frames out of order**  
This should not happen in normal operation. Results are reordered by `job_id` before being returned.
