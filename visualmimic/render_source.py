"""Create a video-only inference manifest and separate preparation metadata."""
import argparse
import json
from pathlib import Path
from .contracts import ArtifactRef
from .storage import ArtifactStore, encode
from .rendering import RenderConfig
from .backends.human_renderer import HumanMeshRenderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Source-only manifest with source, instruction, sample_id')
    p.add_argument('--source-artifacts', required=True)
    p.add_argument('--body-models', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Preparation metadata and source inputs are not in the video directory.
    preparation = ArtifactStore(out / 'preparation', [args.source_artifacts])
    video_store = ArtifactStore(out / 'video/artifacts')
    manifest = json.loads(Path(args.input).read_text())
    if set(manifest) - {'source','instruction','sample_id'}:
        raise ValueError('Unsupported source manifest fields')
    source = preparation.read_motion(ArtifactRef(**manifest['source']))
    config = RenderConfig('fixed-clip-center','neutral-gray','aitviewer-default','white-floor',30,640,480)
    rendered = HumanMeshRenderer(preparation,args.body_models).render(source,config)
    video = video_store.put_bytes(preparation.read_bytes(rendered.video),'video/mp4')
    (out / 'video/manifest.json').write_bytes(encode({'sample_id':manifest['sample_id'],
        'instruction':manifest['instruction'], 'video':video, 'timestamps_s':rendered.timestamps_s,
        'width':config.width,'height':config.height}))
    (out / 'preparation/result.json').write_bytes(encode({'source':manifest['source'],
        'render':rendered,'config':config,'video_manifest':'../video/manifest.json'}))
    print(str(out / 'video/manifest.json'))


if __name__ == '__main__':
    main()
