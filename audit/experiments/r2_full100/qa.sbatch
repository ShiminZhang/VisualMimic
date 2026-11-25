#!/bin/bash
#SBATCH --job-name=vm-full100-video-check
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/ice1/3/5/szhang957/Repos/visualmimic_r2_full100_20261124/qa-%j.log
set -euo pipefail
module load python/3.12.5
cd /storage/ice1/3/5/szhang957/Repos/visualmimic_r2_full100_20261124
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025/.venv_editor/bin/python check_renders.py
