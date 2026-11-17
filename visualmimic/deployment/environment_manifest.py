"""Capture installed versions without importing heavyweight backends."""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PACKAGES = ('numpy','scipy','torch','mujoco','mjlab','mujoco-warp','warp-lang','mink','daqp',
            'smplx','transformers','huggingface-hub','diffusers','hydra-core','omegaconf',
            'pytorch-lightning','torchmetrics','aitviewer','moderngl','imageio','imageio-ffmpeg',
            'joblib','einops','roma','opencv-contrib-python-headless')


def capture():
    versions = {}
    for name in PACKAGES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(capture(),indent=2))
