"""Download a small official MotionFix test bundle; no training set or private assets."""
import hashlib
import json
from pathlib import Path
import gdown

ROOT = Path(__file__).resolve().parents[1] / 'assets'
ASSETS = {
 'tmr/last_weights/motion_encoder.pt': '146ud7aeNvO12LunxPlyoTCq-3TN0KfAg',
 'tmr/config.json': '1QY41LCtgyE8sGCMRNXz2-l1BPfCQ0yOL',
 'tmr/stats/humanml3d/amass_feats/mean.pt': '1fH3vnWTuHcnPIDrZQwUlqb5pWzSJwHN0',
 'tmr/stats/humanml3d/amass_feats/std.pt': '1T0u8b2aIK7go7F1mc7U7i7dXFZxNwwI-',
 'motionfix/motionfix_test.pth.tar': '1wQEEZdsQO-fn43udtWLqwywUeBvTVJPD',
 'motionfix/splits.json': '1n8iMH2z4yPxUj4g8j3loCG7Ny8JVSz5c',
 'motionfix/amt_motionfix_latest.json': '1Yk68qfDdn1kPCkhj27lv-J78Z-3KJ9L_',
 'tmed/.hydra/config.yaml': '19sL2wfV_VCoFaiUcBKD_7YubLt5HOEfx',
 'tmed/checkpoints/last.ckpt': '1V6GnOzNcXaybWz_armNMvhzywXjPCScd',
 'body_models/smplh/SMPLH_NEUTRAL.npz': '1rBRkUNjDLIPh4RLQj0iYDPqY6mGjP3fe',
}


def main():
 ROOT.mkdir(exist_ok=True)
 manifest = ROOT / 'downloads.json'
 completed = json.loads(manifest.read_text()) if manifest.exists() else {}
 for rel, asset_id in ASSETS.items():
  path = ROOT / rel
  path.parent.mkdir(parents=True, exist_ok=True)
  if rel in completed:
   if path.exists() and hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == completed[rel]['sha256']:
    print('VERIFIED', rel, flush=True)
    continue
   raise RuntimeError(f'Previously downloaded asset changed: {rel}')
  if path.exists():
   raise RuntimeError(f'Unrecorded asset exists; inspect before replay: {path}')
  temporary = str(path) + '.download'
  result = gdown.download(id=asset_id, output=temporary, quiet=True, resume=True)
  if not result:
   raise RuntimeError(f'Download did not complete: {rel}')
  Path(temporary).replace(path)
  with path.open('rb') as stream:
   checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
  completed[rel] = {'google_drive_id': asset_id, 'sha256': checksum, 'bytes': path.stat().st_size}
  manifest.write_text(json.dumps(completed, indent=2))
  print('DOWNLOADED', rel, completed[rel], flush=True)


if __name__ == '__main__':
 main()
