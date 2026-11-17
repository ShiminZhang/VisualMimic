"""Coverage-aware orchestration; official metrics must be supplied by a backend."""
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence
from .contracts import ArtifactRef, BackendIdentity, SampleKey, Prediction, EvalReference
from .storage import digest


@dataclass(frozen=True)
class EvaluationProtocol:
    implementation: BackendIdentity
    evaluator_weights: ArtifactRef
    ordered_candidates: ArtifactRef
    motion_conversion_id: str
    generation_and_aggregation_id: str
    expected_keys: tuple[SampleKey, ...]
    seed: int  # one paired replicate per report


class MetricBackend(Protocol):
    identity: BackendIdentity

    def evaluate(self, predictions: Sequence[Prediction], references: Sequence[EvalReference],
                 protocol: EvaluationProtocol) -> dict: ...


def evaluate(results, references, protocol, backend, store):
    """One condition/seed, fixed denominator; missing results remain visible."""
    results, references = list(results), list(references)
    expected = set(protocol.expected_keys)
    if len(expected) != len(protocol.expected_keys) or not expected:
        raise ValueError('Expected sample set must be nonempty and unique')
    if backend.identity != protocol.implementation:
        raise ValueError('Metric backend does not match evaluation protocol')
    store.read_bytes(protocol.evaluator_weights)
    store.read_bytes(protocol.ordered_candidates)
    indexed, instructions = {}, {}
    conditions, comparisons = set(), set()
    fixture = backend.identity.fixture
    for r in results:
        if r.key not in expected or r.key in indexed or r.seed != protocol.seed:
            raise ValueError('Unexpected, duplicate or wrong-seed result')
        if r.status not in {'succeeded', 'failed'}:
            raise ValueError('Invalid result status')
        if (r.status == 'succeeded') != (r.prediction is not None and r.error is None):
            raise ValueError('Inconsistent prediction/status')
        if r.status == 'failed' and (r.error is None or r.prediction is not None):
            raise ValueError('Inconsistent failure record')
        trace = store.read_json(r.trace)
        if trace['status'] != r.status or trace['condition_id'] != r.condition_id or digest(trace['key']) != digest(r.key):
            raise ValueError('Result/trace mismatch')
        if trace['config']['options']['seed'] != r.seed:
            raise ValueError('Trace seed mismatch')
        instructions[r.key] = digest(trace['request']['instruction'])
        fixture |= trace['fixture']
        comparisons.add(trace['comparison_id'])
        conditions.add(r.condition_id)
        if r.prediction:
            p = r.prediction
            if (p.key, p.condition_id, p.seed, p.trace) != (r.key, r.condition_id, r.seed, r.trace):
                raise ValueError('Prediction/result mismatch')
            if digest(p.motion) != digest(trace['artifacts']['prediction']):
                raise ValueError('Prediction/trace artifact mismatch')
            store.read_motion(p.motion)
        indexed[r.key] = r
    if len(conditions) > 1 or len(comparisons) > 1:
        raise ValueError('Mixed experiment conditions or generation settings')
    refs = {r.key: r for r in references}
    if len(refs) != len(references) or set(refs) != expected:
        raise ValueError('References must match the expected set exactly')
    successful = [indexed[k].prediction for k in protocol.expected_keys
                  if k in indexed and indexed[k].status == 'succeeded']
    selected_refs = [refs[p.key] for p in successful]
    for r in selected_refs:
        store.read_bytes(r.source)
        store.read_bytes(r.target)
    metrics = backend.evaluate(successful, selected_refs, protocol) if successful else None
    # Ensure backend results are serializable and finite before publishing.
    digest(metrics)
    failures = Counter(r.error.stage for r in results if r.error)
    missing = [k for k in protocol.expected_keys if k not in indexed]
    return {'protocol_id': digest(protocol), 'comparison_id': next(iter(comparisons), None),
            'condition_id': next(iter(conditions), None), 'fixture': fixture,
            'expected': len(expected), 'succeeded': len(successful),
            'failed': len(results) - len(successful), 'missing': len(missing),
            'coverage': len(successful) / len(expected), 'failure_stages': dict(failures),
            'successful_keys': [p.key for p in successful], 'missing_keys': missing,
            'instruction_hashes': [{'key': k, 'sha256': v} for k, v in instructions.items()],
            'metrics_scope': 'successful_samples', 'metrics': metrics}


def compare_reports(a, b):
    if a['protocol_id'] != b['protocol_id'] or a['comparison_id'] != b['comparison_id']:
        raise ValueError('Reports have incompatible protocols or editing settings')
    if {a['condition_id'], b['condition_id']} != {'source3d', 'video'}:
        raise ValueError('Expected one report for each baseline')
    ia = {digest(r['key']): r['sha256'] for r in a['instruction_hashes']}
    ib = {digest(r['key']): r['sha256'] for r in b['instruction_hashes']}
    if any(ia[k] != ib[k] for k in ia.keys() & ib.keys()):
        raise ValueError('Paired samples have different editing instructions')
    common = set(a['successful_keys']) & set(b['successful_keys'])
    return {'common_successful_keys': sorted(common, key=lambda k: (k.dataset_version, k.split, k.sample_id)),
            'common_successful_count': len(common),
            'requires_common_subset_metric_rerun': set(a['successful_keys']) != set(b['successful_keys'])}
