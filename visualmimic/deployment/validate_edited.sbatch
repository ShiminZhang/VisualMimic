#!/bin/bash
#SBATCH --job-name=visualmimic-edited
#SBATCH --account=def-vganesh_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/s568zhan/SuperMimic-20261024/visualmimic-%j.log
set -euo pipefail
cd /home/s568zhan/SuperMimic-20261024
module load python/3.12.4
unset PYTHONPATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$PWD/.cache"
OUT="evaluation/visualmimic-${SLURM_JOB_ID}/edited"
.venv/bin/python -m visualmimic.execute_reference \
 --robot-motion visualmimic_inputs/motionfix000004_edited_robot.npz \
 --checkpoint logs/lafan1-h100-25068748/ckpt_latest.pt \
 --implementation-commit d9be385b3eeb13a638fb33fb200d3bfd9be9839f \
 --output "$OUT"
.venv/bin/python -m visualmimic.render_execution --run "$OUT" --output "$OUT/rollout.mp4"
