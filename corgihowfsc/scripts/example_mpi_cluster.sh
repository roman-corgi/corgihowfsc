#!/bin/bash
#SBATCH -J RSTMPI               # Job name
#SBATCH -o RSTMPI.o%j           # Name of job output file
#SBATCH -e RSTMPI.e%j           # Name of stderr error file
#SBATCH -p compute             # Queue (partition) name
#SBATCH -N 4                     # Total # of nodes
#SBATCH --cpus-per-task=5       # 5 CPUs per task
#SBATCH --ntasks-per-node=6     # spread evenly across nodes
#SBATCH --mem=256G               # Request about half the RAM on each node
#SBATCH -t 00:10:00            # Run time for 3 iterations (hh:mm:ss)


# Change to work directory
cd /path/to/corgihowfsc/corgihowfsc/scripts

# Load Intel oneAPI into your environment,
# which includes Intel compilers and Intel MPI
source /path/to/oneapi/file.sh

# Set number of threads for each MPI process to spawn 
# TODO : double check if this is the correct setup 
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /path/to/conda.sh
conda activate corgiloop_public

# Run script
mpirun -np $SLURM_NPROCS python run_corgisim_nulling_gitl.py --param_file default_param.yml




