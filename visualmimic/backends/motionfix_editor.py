"""Official TMED inference with a source-only boundary and explicit duration protocol."""
from pathlib import Path
from hashlib import sha256
import sys
import numpy as np
from ..contracts import BackendIdentity, GenerationOptions, MotionSequence
from ..human_motion import HUMAN_SPEC
from ..storage import digest


class MotionFixEditor:
    input_spec = output_spec = HUMAN_SPEC

    def __init__(self, repository, checkpoint, body_models, clip_config, device='cpu',
                 guidance_text=2.0, guidance_motion=2.0):
        import torch
        from omegaconf import OmegaConf
        root = Path(repository).resolve()
        sys.path.insert(0, str(root))
        from src.model.base_diffusion import MD
        from src.data.motionfix import MotionFixDataset
        import smplx
        checkpoint = Path(checkpoint).resolve()
        with checkpoint.open('rb') as stream:
            ck_hash = __import__('hashlib').file_digest(stream, 'sha256').hexdigest()
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        hp = dict(ck['hyper_parameters'])
        hp['statistics_path'] = str(root / 'deps/stats/statistics_motionfix.npy')
        hp['smpl_path'] = str(Path(body_models).resolve())
        hp['text_encoder'] = OmegaConf.create({'_target_': 'visualmimic.backends.checkpoint_clip.CheckpointClipTextEncoder',
                                               'modelpath': str(Path(clip_config).resolve()),
                                               'finetune': False, 'last_hidden_state': True})
        self.model = MD(**hp, renderer=None)
        # Fail if our architecture differs or pretrained text weights are absent.
        state = ck['state_dict']
        for key in ('text_encoder.text_model.text_model.embeddings.position_ids',
                    'text_encoder.text_model.vision_model.embeddings.position_ids'):
            if key in state and key not in self.model.state_dict():
                if not torch.equal(state[key], self.model.get_buffer(key)):
                    raise ValueError(f'CLIP positional indices differ: {key}')
                # Transformers now marks these deterministic buffers nonpersistent.
                del state[key]
        self.model.load_state_dict(state, strict=True)
        del ck
        self.model.to(device).eval().requires_grad_(False)
        self.body_model = smplx.SMPLHLayer(str(Path(body_models) / 'smplh'), gender='neutral', ext='npz').to(device).eval()
        # Stateless use of official feature functions: no dataset/labels loaded here.
        self.feature_reader = object.__new__(MotionFixDataset)
        self.feature_reader.n_body_joints = 22
        self.feature_reader.rot_repr = '6d'
        self.device = device
        self.sampling = {'guidance_text': guidance_text, 'guidance_motion': guidance_motion,
                         'steps': int(self.model.diff_params.num_train_timesteps),
                         'duration': 'source_length', 'source_fps': 30., 'initialization': 'noise',
                         'prob_way': '3way', 'text_preprocessing': 'lowercase'}
        self.sampling_id = digest(self.sampling)
        self.identity = BackendIdentity('motionfix-tmed', 'e6b437e6a17f26b2066f9c142600146f035da6bf+python312-paths',
                                        digest({'sampling': self.sampling,
                                                'stats': sha256((root / 'deps/stats/statistics_motionfix.npy').read_bytes()).hexdigest(),
                                                'body_model': sha256((Path(body_models) / 'smplh/SMPLH_NEUTRAL.npz').read_bytes()).hexdigest()}), ck_hash)

    def options(self, seed=42):
        return GenerationOptions(seed, 'source_length', self.sampling_id)

    def edit(self, source, instruction, options):
        import torch
        from src.tools.transforms3d import transform_body_pose
        source.validate()
        if source.spec != self.input_spec or not source.valid_mask.all():
            raise ValueError('TMED requires canonical complete human motion')
        if not np.allclose(np.diff(source.timestamps_s), 1 / 30., atol=1e-6):
            raise ValueError('Resample explicitly to MotionFix 30 fps before editing')
        if options.duration_policy_id != 'source_length' or options.sampling_config_id != self.sampling_id:
            raise ValueError('Generation configuration mismatch')
        torch.manual_seed(options.seed)
        raw = {'rots': torch.as_tensor(source.fields['rots'].reshape(-1,66)),
               'trans': torch.as_tensor(source.fields['trans']),
               'joint_positions': torch.as_tensor(source.fields['joint_positions'])}
        features = {}
        for name, dim in zip(self.model.input_feats, self.model.input_feats_dims):
            fn = getattr(self.feature_reader, '_get_' + name)
            value = fn(raw)
            if value.shape != (len(source.timestamps_s), dim):
                raise ValueError(f'Official feature shape mismatch: {name}')
            features[name] = value[None].to(self.device)
        with torch.inference_mode():
            normalized = self.model.norm_and_cat_single_motion(features, self.model.input_feats)['motion']
            length = len(source.timestamps_s)
            mask_source, mask_output = self.model.prepare_mot_masks([length], [length], max_len=None)
            generated = self.model.generate_motion([instruction.lower()], normalized, mask_source, mask_output,
                                                    self.model.diffusion_process, gd_text=self.sampling['guidance_text'],
                                                    gd_motion=self.sampling['guidance_motion'],
                                                    num_diff_steps=self.sampling['steps'], show_progress=True)
            decoded = self.model.diffout2motion(generated)[0]
            trans = decoded[:, :3]
            aa = transform_body_pose(decoded[:,3:], '6d->aa').reshape(length,22,3)
            rotations = transform_body_pose(aa.reshape(length,66), 'aa->rot')
            body = self.body_model(global_orient=rotations[:,:1], body_pose=rotations[:,1:], transl=trans)
            result = {'rots': aa.cpu().numpy().astype(np.float32), 'trans': trans.cpu().numpy().astype(np.float32),
                      'joint_positions': body.joints[:,:22].cpu().numpy().astype(np.float32)}
        return MotionSequence(HUMAN_SPEC, source.timestamps_s.copy(), result, source.valid_mask.copy()).validate()
