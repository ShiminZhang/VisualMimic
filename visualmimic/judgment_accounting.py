"""Strict paired-cohort accounting for saved visual-judge outcomes."""
from collections import Counter
from .render_provenance import key_of, unique_index

GROUPS = ('pre_GMR', 'A', 'B', 'C')
VERDICTS = ('success', 'failure', 'uncertain', 'technical_error')


def combine_judgments(tasks, items):
    expected = unique_index(tasks)
    if not expected:
        raise ValueError('Empty judgment cohort')
    indexed = {}
    for item in items:
        key, branch = key_of(item), item['branch']
        if key not in expected or branch not in GROUPS:
            raise ValueError('Unexpected judgment sample/branch')
        if (key, branch) in indexed:
            raise ValueError('Duplicate judgment')
        if item['instruction'] != expected[key]['instruction']:
            raise ValueError('Judgment instruction mismatch')
        assessment = item.get('assessment', {'verdict': 'technical_error', 'reason': item.get('error', '')})
        if not isinstance(assessment, dict) or assessment.get('verdict') not in VERDICTS:
            raise ValueError('Invalid judgment verdict')
        indexed[key, branch] = assessment
    if set(indexed) != {(key, branch) for key in expected for branch in GROUPS}:
        raise ValueError('Missing judgment; full cohort is required')
    rows = [{'key': task['key'], 'instruction': task['instruction'],
             **{branch: indexed[key_of(task), branch] for branch in GROUPS}} for task in tasks]
    counts = {}
    for branch in GROUPS:
        values = Counter(row[branch]['verdict'] for row in rows)
        counts[branch] = {verdict: values[verdict] for verdict in VERDICTS}
    return rows, counts
