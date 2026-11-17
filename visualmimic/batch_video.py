"""Videoize source-only manifests or reconstruct video-only manifests on ICE.

Official sample keys and order are preserved. Preparation metadata is kept out
of the video manifest and reconstruction store. Interrupted calls require an
explicit retry; completed artifacts are verified before reuse.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from .batch_ledger import run_ledger
from .contracts import ArtifactRef, SampleKey, VideoInput
from .storage import ArtifactStore
from .rendering import RenderConfig
from .deployment.environment_manifest import capture, PACKAGES


def validate_rows(rows, stage):
    fields = {'key', 'instruction', 'source'} if stage == 'render' else {
        'key', 'instruction', 'video', 'timestamps_s', 'width', 'height'}
    if stage not in {'render', 'reconstruct'} or not rows:
        raise ValueError('Expected a supported stage and nonempty manifest')
    if any(set(row) != fields for row in rows):
        raise ValueError('Manifest violates source-only/video-only boundary')
    keys = [SampleKey(**row['key']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate sample key')
    return keys


def run_stage(rows, stage, identity, output, process, verify, resume=False,
              retry_interrupted=False):
    validate_rows(rows, stage)
    return run_ledger(rows, stage, identity, output, process, verify, resume, retry_interrupted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['render', 'reconstruct'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--input-artifacts', required=True)
    parser.add_argument('--body-models', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repository')
    parser.add_argument('--python', help='Separate GVHMR environment interpreter')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-interrupted', action='store_true')
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text())
    validate_rows(rows, args.stage)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    # Sibling artifact directories do not create the run before its ledger lock.
    store = ArtifactStore(output.with_name(output.name + '_artifacts'), [args.input_artifacts])
    if args.stage == 'render':
        from .backends.human_renderer import HumanMeshRenderer
        preparation = ArtifactStore(output.with_name(output.name + '_preparation'), [args.input_artifacts])
        backend = HumanMeshRenderer(preparation, args.body_models)
        config = RenderConfig('fixed-clip-center', 'neutral-gray', 'aitviewer-default', 'white-floor', 30, 640, 480)
        identity = {'backend': backend.identity, 'config': config, 'runtime': capture()}
        def process(row):
            source = preparation.read_motion(ArtifactRef(**row['source']))
            rendered = backend.render(source, config)
            video = store.put_bytes(preparation.read_bytes(rendered.video), 'video/mp4')
            preparation.put_json({'key': row['key'], 'source': row['source'], 'render': rendered})
            return {'key': row['key'], 'instruction': row['instruction'], 'video': asdict(video),
                    'timestamps_s': rendered.timestamps_s, 'width': config.width, 'height': config.height}
        def verify(row):
            validate_rows([row], 'reconstruct')
            store.read_bytes(ArtifactRef(**row['video']))
    else:
        if not args.repository or not args.python:
            parser.error('reconstruct requires --repository and --python')
        from .backends.gvhmr_reconstructor import GVHMRReconstructor
        backend = GVHMRReconstructor(store, args.repository, args.python, args.body_models, static_camera=True)
        worker_capture = '''import importlib.metadata as m, json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'packages': versions}))
'''
        worker_runtime = json.loads(subprocess.check_output(
            [backend.python, '-c', worker_capture, json.dumps(PACKAGES)], text=True))
        identity = {'backend': backend.identity, 'runtime': capture(),
                    'worker_python': backend.python, 'worker_runtime': worker_runtime}
        def process(row):
            video = VideoInput(SampleKey(**row['key']), ArtifactRef(**row['video']),
                               tuple(row['timestamps_s']), row['width'], row['height'])
            result = backend.reconstruct(video)
            store.put_json({'key': row['key'], 'reconstruction': result})
            return {'key': row['key'], 'instruction': row['instruction'], 'source': asdict(result.motion)}
        def verify(row):
            validate_rows([row], 'render')
            store.read_motion(ArtifactRef(**row['source']))
    run_stage(rows, args.stage, identity, output, process, verify, args.resume, args.retry_interrupted)


if __name__ == '__main__':
    main()
