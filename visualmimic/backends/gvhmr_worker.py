"""Run official GVHMR preprocessing/inference without optional visualization.

Execute in a separate GVHMR environment, from its repository directory. This
worker accepts only a video; source/target 3-D motions never enter the subprocess.
"""
import sys
from pathlib import Path


def main():
    # The official demo parser accepts --video, --output_root, --static_cam.
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import hydra
    import numpy as np
    from tools.demo.demo import parse_args_to_cfg, run_preprocess, load_data_dict
    from hmr4d.utils.net_utils import detach_to_cpu
    cfg = parse_args_to_cfg()
    run_preprocess(cfg)
    data = load_data_dict(cfg)
    with torch.inference_mode():
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        prediction = detach_to_cpu(model.predict(data, static_cam=cfg.static_cam))
    params = prediction['smpl_params_global']
    output = Path(cfg.output_dir) / 'global_params.npz'
    np.savez_compressed(output, body_pose=params['body_pose'].numpy(),
                        global_orient=params['global_orient'].numpy(),
                        transl=params['transl'].numpy(), betas=params['betas'].numpy())
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
