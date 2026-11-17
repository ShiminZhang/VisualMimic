#!/bin/bash
#SBATCH --job-name=supermimic-h100
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/ice1/3/5/szhang957/Repos/SuperMimic-20261024/train-%j.log
set -euo pipefail
cd /storage/ice1/3/5/szhang957/Repos/SuperMimic-20261024
module load python/3.12.5
unset PYTHONPATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
export TORCHINDUCTOR_COMPILE_THREADS=4
export XDG_CACHE_HOME="$PWD/.cache" TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
test -f PREPARED
nvidia-smi
.venv/bin/python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
assert 'H100' in p.name, p.name
assert p.total_memory > 70 * 1024**3, f'Full H100 required, got {p.total_memory/1024**3:.1f} GiB'
x = torch.randn(1024, 1024, device='cuda')
assert torch.isfinite(x @ x).all()
print('CUDA_PREFLIGHT_OK', p, torch.__version__, flush=True)
PY
exec .venv/bin/sm-train --run-name "lafan1-h100-${SLURM_JOB_ID}" --data.npz-dir data/npz/lafan1 --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 3 --schedule.checkpoint-interval 100 --schedule.log-interval 50
