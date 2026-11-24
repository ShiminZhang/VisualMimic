"""Export saved experiment settings, rendering provenance, and judge records.

Run from the repository root with ``python -m audit.export_ice_evidence ROOT``.
JSON is written to stdout; progress goes to stderr. Original files are read only.
The export omits videos, checkpoints, credentials, API account metadata and
encrypted reasoning. Returned verdict text is preserved exactly.
"""
import hashlib
import json
from pathlib import Path
import sys
from functools import lru_cache

from visualmimic.render_provenance import (
    checked_json, checked_bytes, digest_json, key_of, unique_index,
    verify_pre_gmr_videos,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=None)
def read_file(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def content_identity(value):
    """Compare artifact content rather than machine-specific copy locations."""
    if isinstance(value, dict):
        return {k: content_identity(v) for k, v in value.items() if k != 'uri'}
    if isinstance(value, list):
        return [content_identity(v) for v in value]
    return value


def run_for_artifact(ref):
    directory = Path(ref['uri']).parent
    require(directory.name.endswith('_artifacts'), 'Unrecognized run artifact directory')
    return directory.with_name(directory.name[:-len('_artifacts')])


@lru_cache(maxsize=None)
def run_record(directory):
    value, checksum = read_file(directory / 'run.json')
    require(digest_json(value['specification']) == value['identity'], 'Run identity mismatch')
    return value, checksum, unique_index(value['specification']['rows'])


def stage_settings(row, branch):
    result_ref = row[branch]['result']
    result = checked_json(result_ref)
    require(result['key'] == row['key'] and result['instruction'] == row['instruction'], 'Execution sample mismatch')
    retarget = checked_json(result['retarget_report'])
    require(retarget['key'] == row['key'], 'Retarget sample mismatch')
    require(retarget['robot_motion']['sha256'] == result['input_robot_motion']['sha256'], 'Retarget output mismatch')
    editor_dir = run_for_artifact(retarget['human'])
    editor, editor_sha, editor_rows = run_record(editor_dir)
    item, item_sha = read_file(editor_dir / 'items' / (digest_json(row['key']) + '.json'))
    require(item['status'] == 'succeeded' and item['run_identity'] == editor['identity'], 'Editor item/run mismatch')
    require(item['key'] == row['key'] and item['motion']['sha256'] == retarget['human']['sha256'], 'Editor output mismatch')
    require(editor_rows[key_of(row)]['instruction'] == row['instruction'], 'Editor instruction mismatch')
    ret, ret_sha, ret_rows = run_record(run_for_artifact(result['retarget_report']))
    ret_row = ret_rows[key_of(row)]
    require(ret_row['motion']['sha256'] == retarget['human']['sha256'], 'Retarget run input mismatch')
    require(ret_row['instruction'] == row['instruction'], 'Retarget instruction mismatch')
    execution, execution_sha, execution_rows = run_record(run_for_artifact(result_ref))
    execution_row = execution_rows[key_of(row)]
    require(execution_row['robot_motion']['sha256'] == result['input_robot_motion']['sha256'], 'Execution run input mismatch')
    require(execution_row['retarget_report']['sha256'] == result['retarget_report']['sha256'], 'Execution retarget report mismatch')
    require(execution_row['instruction'] == row['instruction'], 'Execution instruction mismatch')
    metadata = checked_json(result['rollout']['metadata'])
    backend = execution['specification']['backend']
    require(content_identity(backend['policy']) == content_identity(result['policy']) == content_identity(metadata['policy']), 'Saved policy mismatch')
    require(backend['protocol'] == metadata['protocol'], 'Saved execution protocol mismatch')
    require(digest_json(metadata['protocol']) == result['rollout']['protocol_id'], 'Execution protocol hash mismatch')
    require(result['rollout']['policy_id'] == result['policy']['checkpoint']['sha256'], 'Rollout policy hash mismatch')
    require(result['rollout']['reference_id'] == result['reference']['motion_npz']['sha256'], 'Rollout reference mismatch')
    require(backend['render'] == result['render']['config'], 'Render settings mismatch')
    rbackend = ret['specification']['backend']
    settings = {
        'editor': {k: editor['specification'][k] for k in ('editor', 'options', 'execution')},
        'retarget': {k: content_identity(rbackend[k]) for k in ('config', 'mapping', 'robot')},
        'execution': {k: content_identity(backend[k]) for k in ('policy', 'protocol', 'compiler', 'render', 'robot')},
        'initial_reference_frame': metadata['initial_reference_frame'],
    }
    links = {'execution_result_sha256': result_ref['sha256'],
             'execution_metadata_sha256': result['rollout']['metadata']['sha256'],
             'execution_run_sha256': execution_sha, 'retarget_run_sha256': ret_sha,
             'retarget_report_sha256': result['retarget_report']['sha256'],
             'editor_run_sha256': editor_sha, 'editor_item_sha256': item_sha,
             'editor_run_identity': editor['identity'], 'edited_motion_sha256': retarget['human']['sha256']}
    return settings, links, result


def sampled_indices(timestamps):
    count = int((timestamps[-1] - timestamps[0] + 1e-9) / .125) + 1
    selected = {0, len(timestamps) - 1}
    for i in range(count):
        target = timestamps[0] + .125 * i
        selected.add(min(range(len(timestamps)), key=lambda j: abs(timestamps[j] - target)))
    return sorted(selected)


def export(root):
    campaigns = root / 'campaigns'
    combined, combined_sha = read_file(campaigns / 'luna_extend100_v1/combined_report.json')
    retrieval, retrieval_sha = read_file(campaigns / 'abc_g1_retrieval307_v1/report.json')
    require(combined_sha == '10ad60bd0f99f653260b2e4c86cfcd94c033c99b48e22f914de4120ff4a6835a', 'Historical judgment report changed')
    require(retrieval_sha == 'eb287b227dc7dd8af9a4a5c8855fd6417ff81cd945856ef774f6d6a75ae68b64', 'Historical retrieval report changed')
    rows, inputs_sha = read_file(campaigns / 'abc_g1_retrieval307_v1/parent_snapshot/inputs.json')
    require(inputs_sha == retrieval['protocol']['parent_sha256']['inputs.json'], 'Frozen retrieval inputs changed')
    require(len(unique_index(rows)) == 307, 'Expected 307 unique retrieval cases')
    judged = unique_index(combined['items'])
    require(len(judged) == 100 and judged.keys() <= unique_index(rows).keys(), 'Judged/retrieval cohort mismatch')
    configurations, c2, executions = {}, [], {}
    for number, row in enumerate(rows):
        entry = {'key': row['key'], 'judged': key_of(row) in judged}
        executions[key_of(row)] = {}
        for branch in ('A', 'B'):
            settings, links, result = stage_settings(row, branch)
            identity = digest_json(settings)
            configurations[identity] = settings
            entry[branch] = {**links, 'settings_sha256': identity}
            executions[key_of(row)][branch] = result
        require(entry['A']['settings_sha256'] == entry['B']['settings_sha256'], 'A/B settings differ: ' + str(row['key']))
        c2.append(entry)
        if number % 25 == 0: print('C2 verified', number + 1, '/307', file=sys.stderr, flush=True)
    c3 = []
    task_candidates = {}
    first_tasks, _ = read_file(campaigns / 'luna_abc10_v1/tasks.json')
    task_candidates.update(unique_index(first_tasks))
    for name in ('luna_pregmr10_v1', 'luna_extend30_v1', 'luna_extend100_v1'):
        directory = campaigns / name
        tasks, _ = read_file(directory / 'inputs.json')
        videos, _ = read_file(directory / 'render/manifest.json')
        checks = verify_pre_gmr_videos(tasks, videos, directory / 'render_preparation')
        finalized, _ = read_file(directory / 'tasks.json')
        final_index = unique_index(finalized)
        for task, check in zip(tasks, checks):
            k = key_of(task)
            a = executions[k]['A']
            retarget = checked_json(a['retarget_report'])
            require(retarget['human']['sha256'] == check['source_sha256'], 'Pre-GMR is not A GMR input')
            if name != 'luna_pregmr10_v1': task_candidates[k] = final_index[k]
            else: task_candidates[k]['candidates']['pre_GMR'] = final_index[k]['candidates']['pre_GMR']
            candidates = task_candidates[k]['candidates']
            require(candidates['pre_GMR']['video']['sha256'] == check['video_sha256'], 'Finalized pre-GMR video changed')
            require(candidates['C']['source_reference']['motion_npz']['sha256'] == a['reference']['motion_npz']['sha256'], 'C is not A reference')
            checked_bytes(candidates['C']['video'])
            for branch in ('A', 'B'):
                saved = executions[k][branch]
                require(candidates[branch]['source_result']['sha256'] == next(x for x in c2 if key_of(x) == k)[branch]['execution_result_sha256'], 'Candidate execution mismatch')
                require(candidates[branch]['video']['sha256'] == saved['render']['video']['sha256'], 'Candidate rollout video mismatch')
                checked_bytes(candidates[branch]['video'])
            require(candidates['A']['config'] == candidates['B']['config'] == candidates['C']['config'], 'Candidate render configurations differ')
            c3.append({**check, 'A_edited_motion_sha256': retarget['human']['sha256'],
                       'A_reference_sha256': a['reference']['motion_npz']['sha256'],
                       'C_reference_sha256': candidates['C']['source_reference']['motion_npz']['sha256'],
                       'candidate_videos': {g: v['video']['sha256'] for g, v in candidates.items()},
                       'source_video_sha256': task_candidates[k]['source_video']['video']['sha256']})
        print('C3 verified', name, len(checks), file=sys.stderr, flush=True)
    require(unique_index(c3).keys() == judged.keys(), 'Pre-GMR evidence cohort mismatch')
    protocols, c4, checked_sources = {}, [], set()
    directories = [campaigns / name for name in ('luna_abc10_v1', 'luna_pregmr10_v1', 'luna_extend30_v1')]
    directories += [campaigns / 'luna_extend100_v1/shards' / str(i) for i in range(7)]
    for directory in directories:
        protocol, protocol_sha = read_file(directory / 'protocol.json')
        tasks, tasks_sha = read_file(directory / 'tasks.json')
        require(tasks_sha == protocol['source_manifest_sha256'], 'Judge tasks checksum mismatch')
        protocols[protocol_sha] = {'content': protocol, 'source_text': (directory / 'protocol.json').read_text()}
        report, _ = read_file(directory / 'report.json')
        reported = {(key_of(x), x['branch']): x for x in report['items']}
        require(len(reported) == len(report['items']), 'Duplicate judge report item')
        for request_dir in sorted((directory / 'requests').iterdir()):
            request, request_record_sha = read_file(request_dir / 'request.json')
            result, result_sha = read_file(request_dir / 'result.json')
            response, response_sha = read_file(request_dir / 'response.json')
            k, branch = key_of(result), result['branch']
            require((k, branch) in reported and reported[k, branch] == result, 'Shard/result mismatch')
            require(result['status'] == 'assessed' and response['status'] == 'completed', 'Unassessed historical request')
            require(response['model'].startswith('gpt-6-luna'), 'Unexpected judge model')
            for field in ('key', 'instruction', 'branch', 'source', 'candidate', 'source_frame_indices', 'candidate_frame_indices', 'request_sha256'):
                require(request[field] == result[field], 'Request/result metadata mismatch: ' + field)
            texts = [c['text'] for item in response['output'] for c in item.get('content', []) if c.get('type') == 'output_text']
            raw_text = ''.join(texts)
            assessment = json.loads(raw_text)
            require(assessment == result['assessment'] == judged[k][branch], 'Response/summary verdict mismatch')
            require(assessment['verdict'] in ('success', 'failure', 'uncertain'), 'Invalid judge verdict')
            source = task_candidates[k]['source_video']
            candidate = task_candidates[k]['candidates'][branch]
            if k not in checked_sources:
                checked_bytes(source['video'])
                checked_sources.add(k)
            require(request['source']['sha256'] == source['video']['sha256'], 'Judge source video mismatch')
            require(request['candidate']['sha256'] == candidate['video']['sha256'], 'Judge candidate video mismatch')
            require(request['source_frame_indices'] == sampled_indices(source['timestamps_s']), 'Source frame selection mismatch')
            require(request['candidate_frame_indices'] == sampled_indices(candidate['timestamps_s']), 'Candidate frame selection mismatch')
            c4.append({'key': result['key'], 'branch': branch, 'instruction': result['instruction'],
                       'source_video_sha256': request['source']['sha256'], 'candidate_video_sha256': request['candidate']['sha256'],
                       'source_frame_indices': request['source_frame_indices'], 'candidate_frame_indices': request['candidate_frame_indices'],
                       'request_sha256': request['request_sha256'], 'request_record_sha256': request_record_sha,
                       'response_sha256': response_sha, 'result_sha256': result_sha,
                       'protocol_sha256': protocol_sha, 'response_model': response['model'],
                       'verdict': assessment['verdict'], 'output_text': raw_text})
        print('C4 verified', directory.relative_to(campaigns), file=sys.stderr, flush=True)
    require(len(c4) == len({(key_of(x), x['branch']) for x in c4}) == 400, 'Expected 400 unique judgments')
    return {'schema_version': 1, 'source_reports': {'judgments': combined_sha, 'retrieval': retrieval_sha},
            'c2': {'configurations': configurations, 'items': c2}, 'c3': c3,
            'c4': c4, 'protocols': protocols,
            'limitations': ['No model inference, policy training, rendering or judge calls were repeated.',
                            'Settings are verified from saved run identities and linked per-case records; this does not establish bitwise replay.',
                            'Full API image payloads were not archived; recorded request hashes were cross-checked, not reconstructed.',
                            'API envelopes and encrypted reasoning remain on ICE; exact returned verdict text and envelope hashes are exported.',
                            'Video hashes and preparation associations were checked; this is not human validation of instruction success.']}


if __name__ == '__main__':
    print(json.dumps(export(Path(sys.argv[1]).resolve()), sort_keys=True, separators=(',', ':'), allow_nan=False))
