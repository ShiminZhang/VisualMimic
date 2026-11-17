"""Strict numerical diagnostic, not a replacement for official editing metrics."""
import numpy as np


def compare_reconstruction(reconstructed, source):
    reconstructed.validate()
    source.validate()
    if reconstructed.spec != source.spec:
        raise ValueError('Diagnostics require explicitly matched representations')
    if not np.array_equal(reconstructed.timestamps_s, source.timestamps_s):
        raise ValueError('Diagnostics require explicit temporal alignment')
    mask = reconstructed.valid_mask & source.valid_mask
    if not mask.any():
        raise ValueError('No jointly valid frames')
    return {'alignment': 'none', 'units': source.spec.units, 'frames': int(mask.sum()),
            'total_frames': len(mask), 'per_field_rmse': {
                name: float(np.sqrt(np.mean((reconstructed.fields[name][mask].astype(float)
                                             - source.fields[name][mask].astype(float)) ** 2)))
                for name in source.fields}}
