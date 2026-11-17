"""Official MotionFix motion features and TMR encoder for human-layer diagnostics.

Only the motion encoder is needed for motion-to-motion retrieval. Learned weights
are loaded strictly, and VAE mean encoding matches the official evaluator.
"""
import json
import sys
from pathlib import Path
from hashlib import file_digest
import numpy as np
from ..contracts import BackendIdentity
from ..human_motion import HUMAN_SPEC
from ..storage import digest


class TMRMotionEvaluator:
    def __init__(self, repository, assets, device='cpu'):
        import torch
        root, assets = Path(repository).resolve(), Path(assets).resolve()
        sys.path.insert(0, str(root))
        from src.tmr.actor import ACTORStyleEncoder
        config = json.loads((assets / 'config.json').read_text())
        parameters = dict(config['model']['motion_encoder'])
        parameters.pop('_target_')
        self.encoder = ACTORStyleEncoder(**parameters)
        weights = assets / 'last_weights/motion_encoder.pt'
        self.encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.device = device
        stats = assets / 'stats/humanml3d/amass_feats'
        self.mean = torch.load(stats / 'mean.pt', weights_only=True, map_location='cpu')
        self.std = torch.load(stats / 'std.pt', weights_only=True, map_location='cpu')
        if self.mean.shape != (135,) or self.std.shape != (135,) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std < 0).any():
            raise ValueError('Invalid official motion feature statistics')
        hashes = {}
        for path in (weights, stats/'mean.pt', stats/'std.pt', assets/'config.json'):
            with path.open('rb') as stream:
                hashes[str(path.relative_to(assets))] = file_digest(stream, 'sha256').hexdigest()
        self.identity = BackendIdentity('motionfix-tmr-motion', 'e6b437e6a17f26b2066f9c142600146f035da6bf',
                                        digest({'assets':hashes,'encoding':'vae-mean','features':'official-135'}),
                                        hashes['last_weights/motion_encoder.pt'])

    def features(self, motion):
        import torch
        from src.data.features import _get_body_pose, _get_body_orient, _get_body_transl_delta_pelv_infer
        motion.validate()
        if motion.spec != HUMAN_SPEC or not motion.valid_mask.all():
            raise ValueError('TMR requires complete canonical human motion')
        if len(motion.timestamps_s) < 2 or not np.allclose(np.diff(motion.timestamps_s),1/30.,atol=1e-6):
            raise ValueError('Explicit 30 fps motion required')
        rots = torch.from_numpy(motion.fields['rots'].reshape(-1,66))
        orient = _get_body_orient(rots[:,:3])
        delta = _get_body_transl_delta_pelv_infer(orient, torch.from_numpy(motion.fields['trans']))
        value = torch.cat([delta, _get_body_pose(rots), orient], dim=-1)
        return (value - self.mean) / (self.std + 1e-12)

    def encode(self, motions, batch_size=16):
        import torch
        from src.data.tools.collate import collate_tensor_with_padding
        if not motions or batch_size < 1:
            raise ValueError('Nonempty motions and positive batch size required')
        embeddings = []
        with torch.inference_mode():
            for start in range(0,len(motions),batch_size):
                features = [self.features(m) for m in motions[start:start+batch_size]]
                lengths = torch.tensor([len(f) for f in features], device=self.device)
                x = collate_tensor_with_padding(features).to(self.device)
                mask = torch.arange(x.shape[1],device=self.device)[None] < lengths[:,None]
                # Token zero is mu; token one is log variance (official sample_mean=True).
                encoded = self.encoder({'x':x,'mask':mask})[:,0]
                embeddings.append(encoded.cpu().numpy())
        return np.concatenate(embeddings)

    def paired_retrieval(self, ordered_ids, reference_motions, generated_motions):
        """Official all-set direction: reference queries against generated candidates.

        Caller supplies a fixed, complete, ordered evaluation set. No silent removal
        of failed samples and no substitution of a smaller candidate set is allowed.
        """
        from src.tmr.metrics import all_contrastive_metrics_mot2mot
        if len(set(ordered_ids)) != len(ordered_ids) or len(ordered_ids) < 2:
            raise ValueError('Retrieval requires at least two unique fixed candidates')
        if len(ordered_ids) != len(reference_motions) or len(ordered_ids) != len(generated_motions):
            raise ValueError('Missing predictions: preserve the fixed candidate set')
        # Official evaluator crops each generated motion to its own target length.
        from ..contracts import MotionSequence
        cropped = []
        for target, generated in zip(reference_motions,generated_motions):
            count = min(len(target.timestamps_s),len(generated.timestamps_s))
            cropped.append(MotionSequence(generated.spec, generated.timestamps_s[:count],
                {k:v[:count] for k,v in generated.fields.items()},generated.valid_mask[:count]))
        a,b = self.encode(reference_motions),self.encode(cropped)
        a /= np.linalg.norm(a,axis=1,keepdims=True)
        b /= np.linalg.norm(b,axis=1,keepdims=True)
        similarities = a @ b.T
        return {'backend':self.identity,'ordered_ids':list(ordered_ids),'candidates_id':digest(ordered_ids),
                'direction':'reference-to-generated', 'protocol':'official-all-set-retrieval',
                'length_policy':'generated-cropped-to-own-reference-length',
                'metrics':all_contrastive_metrics_mot2mot(similarities),
                'similarities':similarities.tolist(),
                'scope':'human-motion retrieval; not robot execution or instruction success'}
