"""Render a completed execution without replaying the policy or physics."""
import argparse
import json
from pathlib import Path
from .contracts import ArtifactRef
from .robot_contracts import RobotRollout
from .rollout_rendering import render_rollout
from .storage import ArtifactStore, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Directory containing result.json and artifacts/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-xml', help='Optional portable copy of the exact execution model XML and meshes')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    result = json.loads((run / 'result.json').read_text())
    raw = result['rollout'].copy()
    store = ArtifactStore(run / 'artifacts')
    # Content hashes allow downloaded runs to move without trusting old absolute paths.
    for key in ('trajectory', 'events', 'metadata'):
        ref = raw[key].copy()
        ref['uri'] = str(store.root / ref['sha256'])
        raw[key] = ArtifactRef(**ref)
    rollout = RobotRollout(**raw)
    video = render_rollout(rollout, store, args.output, model_xml=args.model_xml)
    manifest = Path(args.output).with_suffix('.json')
    manifest.write_bytes(encode(video))
    print(str(manifest))


if __name__ == '__main__':
    main()
