#!/bin/bash
#SBATCH -J mpi
#SBATCH --nodes=4               # 4 nodes (32 cores per node)
#SBATCH --ntasks=22             # num workers + 1 master
#SBATCH --ntasks-per-node=6     # spread evenly across nodes
#SBATCH -t 24:00:00             # wall time limit (hh:mm:ss)
#SBATCH --cpus-per-task=5       # 5 CPUs per task (for PROPER)
#SBATCH --mem=45G
#SBATCH -o mpi_test_%A.txt
#SBATCH -e mpi_test_error_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexis.lau@lam.fr

module purge
module load compiler/2024.2.1
module load mpi/2021.13

source ~/.bashrc
conda activate corgiloop

# python 
mpirun -np $SLURM_NTASKS python run_corgisim_nulling_gitl.py --param_file default_param.yml
