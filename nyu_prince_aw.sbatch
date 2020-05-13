#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=64000
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, train_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Debug
echo $PATH
export PATH=/beegfs/aw3272/software/miniconda3/envs/tfmr/lib:$PATH
export LD_LIBRARY_PATH=/beegfs/aw3272/software/miniconda3/envs/tfmr/lib:$LD_LIBRARY_PATH

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - ${job_name} >> ~/jiant_machine_assignments.txt

# Run.
cd ~/projects/transformers
./run.sh train ${task} 0 ${seed} ${lr} ${bz}
