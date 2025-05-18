#!/bin/bash

#SBATCH -J debug-slurm
#SBATCH -t 0:05:00
#SBATCH --mem=1G
#SBATCH -o "/hps/nobackup/birney/users/sfan/logs/%j.out"
#SBATCH -e "/hps/nobackup/birney/users/sfan/logs/%j.out"

echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"
