#!/bin/bash
#SBATCH --job-name=vm-stage-judge
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/storage/ice1/3/5/szhang957/Repos/visualmimic_r2_20261124/judge-%j.log
set -euo pipefail
module load python/3.12.5
cd /storage/ice1/3/5/szhang957/Repos/visualmimic_r2_20261124
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$PWD/cache" MPLCONFIGDIR="$PWD/cache/matplotlib"
/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025/.venv_editor/bin/python experiment.py judge
/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025/.venv_editor/bin/python experiment.py summarize
