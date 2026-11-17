#!/usr/bin/env bash
# Stage the complete project on ICE without deleting local or remote files.
# Run from PARL-VLM after ICE SSH access is restored.
set -euo pipefail
ICE_HOST="${VISUALMIMIC_ICE_HOST:-szhang957@login-ice.pace.gatech.edu}"
ICE_ROOT="${VISUALMIMIC_ICE_ROOT:-/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025}"
[[ "$ICE_ROOT" =~ ^/storage/ice1/[A-Za-z0-9_./-]+$ ]] || { echo 'Unexpected ICE root'; exit 1; }
[[ "$ICE_HOST" =~ ^[A-Za-z0-9@._-]+$ ]] || { echo 'Unexpected SSH host'; exit 1; }
test -f VisualMimic.md
test -f visualmimic/batch_edit.py
SSH='ssh -o BatchMode=yes -o ConnectTimeout=15'
$SSH "$ICE_HOST" "set -eu; if test -d '$ICE_ROOT'; then test -f '$ICE_ROOT/.visualmimic-migration'; else mkdir '$ICE_ROOT'; touch '$ICE_ROOT/.visualmimic-migration'; fi; mkdir -p '$ICE_ROOT/archive/local_outputs' '$ICE_ROOT/tests' '$ICE_ROOT/policies'; df -h '$ICE_ROOT'"
rsync -az --partial --partial-dir=.rsync-partial -e "$SSH" --exclude __pycache__ visualmimic/ "$ICE_HOST:$ICE_ROOT/visualmimic/"
rsync -az --partial --partial-dir=.rsync-partial -e "$SSH" --exclude __pycache__ supermimic-reproduction/ "$ICE_HOST:$ICE_ROOT/supermimic-reproduction/"
rsync -az --partial --partial-dir=.rsync-partial -e "$SSH" outputs/visualmimic/ "$ICE_HOST:$ICE_ROOT/archive/local_outputs/"
rsync -az -e "$SSH" VisualMimic.md "$ICE_HOST:$ICE_ROOT/"
rsync -az -e "$SSH" tests/test_visualmimic*.py "$ICE_HOST:$ICE_ROOT/tests/"
CHECKPOINT=outputs/visualmimic/integration/execution_source/artifacts/387e32872046d7e7a618322dc9c832e1f021cf2aa389c3407143718075bed1e2
if test -f "$CHECKPOINT"; then
 rsync -az --partial -e "$SSH" "$CHECKPOINT" "$ICE_HOST:$ICE_ROOT/policies/supermimic-lafan1-39100.pt"
fi
# Checksum verification is a separate non-mutating pass. Nonempty output needs review.
rsync -aznc -e "$SSH" --exclude __pycache__ --out-format='%n' visualmimic/ "$ICE_HOST:$ICE_ROOT/visualmimic/"
rsync -aznc -e "$SSH" --out-format='%n' outputs/visualmimic/ "$ICE_HOST:$ICE_ROOT/archive/local_outputs/"
echo "Transferred to $ICE_HOST:$ICE_ROOT. Recreate prepared manifests on ICE to use native paths. No jobs were submitted."
