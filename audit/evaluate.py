"""Validate and aggregate the released verdicts and composite retrieval ranks.

This checks exported records. It does not rerun a visual judge or motion model.
All verdict categories remain in the denominator, including zero-count groups.
"""
from pathlib import Path
from collections import Counter
import csv
import math
from statistics import median

STAGES = ('pre_GMR', 'C', 'A', 'B')
VERDICTS = ('success', 'failure', 'uncertain', 'technical_error')


def read_table(path, fields, expected_rows):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(fields) or set(reader.fieldnames) != set(fields):
            raise ValueError(f'Invalid columns in {path}')
        rows = list(reader)
    if len(rows) != expected_rows or not rows:
        raise ValueError(f'Expected {expected_rows} rows in {path}')
    if any(set(row) != set(fields) or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed row in {path}')
    ids = [row['sample_id'] for row in rows]
    if any(not value.strip() or value != value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Empty or duplicate sample ID in {path}')
    return rows


def load_tables(root, expected_judgments=100, expected_retrieval=307):
    root = Path(root)
    verdicts = read_table(root / 'judgments.csv', ('sample_id', *STAGES), expected_judgments)
    retrieval = read_table(root / 'composite_ranks.csv', ('sample_id', 'A', 'B', 'C'), expected_retrieval)
    for row in verdicts:
        if any(row[stage] not in VERDICTS for stage in STAGES):
            raise ValueError('Unknown or missing verdict')
    for row in retrieval:
        for stage in ('A', 'B', 'C'):
            rank = float(row[stage])
            if not math.isfinite(rank) or not rank.is_integer() or not 1 <= rank <= len(retrieval):
                raise ValueError('Rank must be an integer within the candidate set')
    if not {row['sample_id'] for row in verdicts} <= {row['sample_id'] for row in retrieval}:
        raise ValueError('Judged sample absent from retrieval cohort')
    return verdicts, retrieval


def summarize(verdicts, retrieval):
    result = {}
    for stage in STAGES:
        counts = Counter(row[stage] for row in verdicts)
        result[stage] = {'judged': len(verdicts), **{label: counts[label] for label in VERDICTS}}
        if stage != 'pre_GMR':
            ranks = [float(row[stage]) for row in retrieval]
            result[stage].update({'retrieval_queries': len(ranks),
                'R@1': 100 * sum(rank <= 1 for rank in ranks) / len(ranks),
                'R@5': 100 * sum(rank <= 5 for rank in ranks) / len(ranks), 'MedR': median(ranks)})
    return result


def main():
    tables = load_tables(Path(__file__).parent)
    for stage, metrics in summarize(*tables).items():
        print(f"{stage}: {metrics['success']}/{metrics['judged']} successes")
        print('  ' + ', '.join(f'{label}={metrics[label]}' for label in VERDICTS))
        if stage != 'pre_GMR':
            print(f"  R@1: {metrics['R@1']:.2f}%; R@5: {metrics['R@5']:.2f}%; Median rank: {metrics['MedR']:g} ({metrics['retrieval_queries']} queries)")


if __name__ == '__main__':
    main()
