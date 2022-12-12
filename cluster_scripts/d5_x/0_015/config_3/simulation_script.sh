#!/bin/bash
#SBATCH -p scavenger                         # scavenger division
#SBATCH -c 1                                 # Number of cores
#SBATCH --array=1-1  			             # How many jobs do you have                               
#SBATCH --job-name=0.015_3          # Job name, will show up in squeue output
#SBATCH --mail-type=END
#SBATCH --mail-user=ql94@duke.edu	       # It will send you an email when the job is finished. 
#SBATCH --mem=1G                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output=out.out         # File to which standard out will be written
#SBATCH --error=slurm.err           # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

#module load python/3.6.5
#module load tensorflow/1.14.0
#module load keras/1.14.0

# ------- run the script -----------------------

python /hpc/home/ql94/DeepQ-Decoding/cluster_scripts/d5_x/0_015/Single_Point_Continue_Training_Script.py 3

#----------- wait some time ------------------------------------

sleep 50