#!/bin/bash
#SBATCH --job-name=supermimic-prepare
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/ice1/3/5/szhang957/Repos/SuperMimic-20261024/prepare-%j.log
set -euo pipefail
cd /storage/ice1/3/5/szhang957/Repos/SuperMimic-20261024
module load python/3.12.5
unset PYTHONPATH
export PYTHONUNBUFFERED=1 UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface"
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 OMP_NUM_THREADS=1
python3 -m venv .bootstrap
.bootstrap/bin/pip install uv
.bootstrap/bin/uv sync --locked --group dev
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
repo = 'lvhaidong/LAFAN1_Retargeting_Dataset'
info = HfApi().dataset_info(repo)
open('data_revision.txt', 'w').write(info.sha + '\n')
snapshot_download(repo, repo_type='dataset', revision=info.sha,
                  allow_patterns=['g1/*.csv'], local_dir='data/raw/lafan1', max_workers=4)
PY
.venv/bin/python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30 --workers 8
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(open('data/npz/lafan1/manifest.csv')))
assert len(rows) == 40, f'Expected 40 LAFAN1 motions, got {len(rows)}'
assert all(float(r['fps']) == 50 for r in rows)
print('DATA_READY', len(rows), sum(float(r['duration_s']) for r in rows) / 3600, flush=True)
Path('PREPARED').write_text('40 LAFAN1 motions converted to 50 Hz\n')
PY
