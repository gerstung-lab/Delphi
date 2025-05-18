#!/bin/bash

#SBATCH -J delphi-prs
#SBATCH -t 3:00:00
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH -o "/hps/nobackup/birney/users/sfan/logs/%j.out"
#SBATCH -e "/hps/nobackup/birney/users/sfan/logs/%j.out"

source ~/.bashrc

set -a
source submit/ebi/var.env
set +a

module load cuda/11.8.0

micromamba activate delphi-cf-torch2.3

echo $DELPHI_DATA_DIR

echo "First argument: $1"
echo "Second argument: $2"
echo "All arguments: $@"

python apps/train.py config=config/override/train.yaml device=cuda $1
