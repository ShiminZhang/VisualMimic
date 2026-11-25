#!/bin/bash
#SBATCH --job-name=vm-stage-render
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/ice1/3/5/szhang957/Repos/visualmimic_r2_20261124/render-%j.log
set -euo pipefail
module load python/3.12.5
cd /storage/ice1/3/5/szhang957/Repos/visualmimic_r2_20261124
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$PWD/cache" MPLCONFIGDIR="$PWD/cache/matplotlib" TORCH_HOME="$PWD/cache/torch"
/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025/.venv_editor/bin/python experiment.py human
/storage/ice1/3/5/szhang957/Repos/SuperMimic-20261024/.venv/bin/python experiment.py robot
