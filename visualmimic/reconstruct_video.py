"""Reconstruct a video-only manifest into the canonical editor input.

The output source_manifest.json can be consumed by prepare_robot_pair with this
run's artifacts directory. Inference input excludes preparation camera and 3-D data.
"""
import argparse
import json
from pathlib import Path
from .contracts import ArtifactRef, SampleKey, VideoInput
from .storage import ArtifactStore, encode
from .backends.gvhmr_reconstructor import GVHMRReconstructor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--video-artifacts', required=True)
    p.add_argument('--dataset-version', required=True)
    p.add_argument('--split', choices=['train','val','test'], required=True)
    p.add_argument('--repository', required=True)
    p.add_argument('--python', required=True, help='Python in the separate GVHMR GPU environment')
    p.add_argument('--body-models', required=True, help='SMPL-H assets for the canonical editor representation')
    p.add_argument('--moving-camera', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    store = ArtifactStore(out/'artifacts',[args.video_artifacts])
    status = out/'status.json';status.write_bytes(encode({'status':'running','arguments':vars(args)}))
    try:
        manifest = json.loads(Path(args.input).read_text())
        if set(manifest) != {'sample_id','instruction','video','timestamps_s','width','height'}:
            raise ValueError('Expected video-only manifest; source/target and camera metadata are forbidden')
        video = VideoInput(SampleKey(args.dataset_version,manifest['sample_id'],args.split),
            ArtifactRef(**manifest['video']),tuple(manifest['timestamps_s']),manifest['width'],manifest['height'])
        backend = GVHMRReconstructor(store,args.repository,args.python,args.body_models,
                                      static_camera=not args.moving_camera)
        result = backend.reconstruct(video)
        (out/'result.json').write_bytes(encode({'backend':backend.identity,'video':video,'reconstruction':result}))
        (out/'source_manifest.json').write_bytes(encode({'sample_id':manifest['sample_id'],
            'instruction':manifest['instruction'],'source':result.motion}))
        status.write_bytes(encode({'status':'reconstructed','editing_and_execution':'not_run'}))
        print(str(out/'source_manifest.json'))
    except Exception as exc:
        status.write_bytes(encode({'status':'failed','error':str(exc)}))
        raise


if __name__ == '__main__':
    main()
