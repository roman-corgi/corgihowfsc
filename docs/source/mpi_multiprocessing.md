# Parallel Runs: Local and Multi-Node MPI

This page explains how to choose and size parallel runs in `corgihowfsc`.

`corgihowfsc` supports two ways to run in parallel: 

- **Local run**: run on one machine using multiprocessing. 
- **Multi-node (MPI) run**: runs with MPI ranks, usually on a cluster, across multiple nodes. 

For the full YAML field descriptions, see [Parameter Reference](parameter_reference.md).
For details on how the local multiprocessing and MPI setups are arranged, see [Parallel Setup Details](parallel_setup_details.md).


---

### Quick Start


### Local Run

Use local mode on a laptop, workstation, or single cluster node.

:::{note}
The values below are examples. Choose `num_imager_worker`, `num_proper_process`, and `num_jac_process` based on the CPUs available on your machine. See [Sizing a Local Run](#sizing-a-local-run).
:::

```yaml
runtime:
  use_mpi: false
  num_imager_worker: 3  # null: serial image generation; set > 1 for local image workers
  num_proper_process: 5    # PROPER processes inside each image worker
  num_jac_process: 6       # local Jacobian processes
```


Run:

```bash
python run_corgisim_nulling_gitl.py --param_file default_param.yml
```

### Multi-node (MPI) Run

Use an MPI run when one machine is not enough and you want to spread the simulation across multiple cluster nodes. **MPI (Message Passing Interface)** is a standard communication protocol for parallel programs that coordinates separate processes across a cluster, including across nodes with separate memory, by passing messages between them.

In an MPI run, one Python process acts as the manager. It runs the GITL loop and sends image generation or Jacobian calculations to worker processes. The worker processes wait for tasks, run them, and send the results back to the manager. 

This is useful because CorgiSim/PROPER image generation and Jacobian computation can be slow and resource intensive. MPI lets several workers run these tasks at the same time across a cluster allocation.

MPI documentation often calls these processes "ranks". This page uses "manager process" and "worker processes" to explain the run. 

```yaml
runtime:
  use_mpi: true
  num_imager_worker: 21    # MPI worker processes
  num_proper_process: 5    # PROPER processes per worker process
  num_jac_process: 6       # ignored in MPI mode
```

For this example, we launch 22 total MPI processes: 

```text
1 manager process + 21 worker processes = 22 total MPI processes
```

Example Slurm launch:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=5

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun -np $SLURM_NTASKS python run_corgisim_nulling_gitl.py --param_file example_mpi_param.yml
```

:::{note}
Example MPI submission scripts are available in `corgihowfsc/scripts/example_mpi_*.sh`. Use them as templates, but check the module loads, conda environment, `--ntasks`, and `--cpus-per-task` values for your cluster.
:::


---

## Choose Between Local and MPI 

| Situation | Recommended Run Type |
| --- | --- |
| Laptop or workstation | Local run |
| First test run or debugging | Local run |
| Single-node run with enough CPUs | Local run |
| Multi-node cluster allocation | MPI run |
| Slow CorgiSim image generation across many iterations | MPI run |

Start with a local run. Switch to MPI when the per-iteration wall time is the bottleneck and you have a cluster allocation available.

---

## Parallel Parameters

| Parameter | Local Run (`use_mpi: false`) | MPI Run (`use_mpi: true`) |
| --- | --- | --- |
| `num_imager_worker` | Number of local image worker processes. `null` means serial/default. | Number of MPI worker processes. Usually `--ntasks - 1`. |
| `num_proper_process` | Number of PROPER subprocesses inside each local image worker. | Number of PROPER subprocesses inside each MPI worker process. Usually matches `--cpus-per-task`. |
| `num_jac_process` | Number of local Jacobian processes. | Ignored. MPI Jacobian chunks use the MPI worker processes capped by `num_imager_worker`. |
| `use_mpi` | `false` | `true` |

---

## Sizing a Local Run

For local image generation, the approximate peak CPU use is:

```text
num_imager_worker x num_proper_process
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

`num_jac_process` controls a separate local Jacobian process pool. It is not active at the exact same time as image generation, but it should still be no larger than the CPUs available for the job.

At the start of the local run, it checks for oversubscription and warns if the requested process count exceeds the CPUs visible to the process. Oversubscription means asking for more parallel processes than available CPU cores, which can make the run slower rather than faster because processes compete for the same cores.

For a first local run, start conservatively:

```yaml
runtime:
  use_mpi: false
  num_imager_worker: null
  num_proper_process: 5
  num_jac_process: 6
```

Increase `num_imager_worker` only if image generation is the bottleneck and the machine has enough CPUs for the additional workers.

---

## MPI Run Model

In MPI mode, one process is the manager. It runs the main loop and sends work to the workers.

```text
process 0       manager
process 1..N-1  workers
```

Worker processes are initialized once and reused for image generation and Jacobian chunks.

The manager process does not compute image or Jacobian tasks. Therefore:

```text
total MPI processes = num_imager_worker + 1
```

:::{important}
In MPI mode, launch `num_imager_worker + 1` total MPI processes. Rank 0 is the manager and does not run frame or Jacobian tasks.
:::

Example:

```yaml
num_imager_worker: 21
```

requires:

```bash
mpirun -np 22 ...
```

or in Slurm:

```bash
#SBATCH --ntasks=22
```

---

## MPI Process and CPU Sizing

For MPI mode, use this mapping:

```text
--ntasks         = num_imager_worker + 1
--cpus-per-task  = num_proper_process
```

Example YAML:

```yaml
runtime:
  use_mpi: true
  num_imager_worker: 21
  num_proper_process: 5
```

Example Slurm settings:

```bash
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=5
```

This means:

```text
1 manager process
21 worker processes
5 CPUs available to each MPI process
```

Each worker process may spawn up to `num_proper_process` PROPER subprocesses.

---

## Choosing `num_imager_worker`

For image generation, the useful maximum is the number of images in one framelist:

```text
n_wvl x (2 x n_probe_pairs + 1)
```

For HLC band 1 with 3 wavelengths and 3 probe pairs:

```text
3 x (2 x 3 + 1) = 21
```

So a natural MPI setup is:

```yaml
num_imager_worker: 21
```

and:

```bash
#SBATCH --ntasks=22
```

Using more than 21 worker processes will not speed up image generation for this case because there are only 21 image tasks in one framelist. Extra workers may still be initialized, but they will sit idle during image collection.

---

## Process-Per-Core Mode

The intended parallel model is process-based:

```text
MPI processes x PROPER processes
```

Threaded numerical libraries should be kept single-threaded because this workflow uses process-based parallelism:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

PROPER's multi-run path uses a multiprocessing pool for `NCPUS` work and sets its multi-process FFT thread count to 1 internally. The environment variables above prevent threaded math libraries from adding extra parallelism around the process-based model.

Do not use:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

---

## Slurm Launch Notes

Prefer an explicit MPI process count:

```bash
mpirun -np $SLURM_NTASKS python run_corgisim_nulling_gitl.py --param_file example_mpi_param.yml
```

On systems where `mpirun` correctly uses the scheduler allocation, you can also omit `-np`:

```bash
mpirun python run_corgisim_nulling_gitl.py --param_file mpi_param.yml
```

:::{warning}
Check that the MPI process count launched by Slurm matches `num_imager_worker + 1`. Extra MPI processes are initialized but remain idle; too few MPI processes reduce the requested parallelism.
:::

If you use:

```bash
#SBATCH --ntasks-per-node=6
```

some MPI launchers may infer the total process count from:

```text
nodes x ntasks-per-node
```

For example:

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=6
```

This can launch 24 MPI processes. If your YAML has:

```yaml
num_imager_worker: 21
```

you only wanted 22 total MPI processes, so 2 worker processes will be initialized but idle.

---

## Common Startup Warnings

### More MPI Processes Than Requested Workers

```text
MPI launched 23 worker processes, but num_imager_worker=21.
Extra worker processes will be initialized but remain idle.
```

This means MPI launched more processes than your YAML will use. The run is still correct, but some allocated processes are wasted.

Fix by launching exactly:

```text
num_imager_worker + 1
```

total MPI processes.

### Fewer MPI Processes Than Requested Workers

```text
MPI requested 21 active worker processes, but only 9 worker processes were launched.
The MPI task queues will use 9 workers.
```

The run is still correct, but slower than expected.

Fix by increasing the MPI process count or reducing `num_imager_worker`.

### PROPER Processes Exceed CPUs Per Task

```text
num_proper_process exceeds the CPU affinity visible to the MPI process
```

This usually means:

```yaml
num_proper_process
```

is larger than:

```bash
#SBATCH --cpus-per-task
```

Fix by making them match, or by reducing `num_proper_process`.
