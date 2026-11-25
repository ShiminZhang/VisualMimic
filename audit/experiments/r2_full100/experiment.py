"""Matched evaluation of saved motion-editing stages in an isolated directory."""
import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025')
OUT = Path(__file__).resolve().parent
CONDITIONS = ('A_human', 'B_human', 'A_reference', 'B_reference', 'A_execution', 'B_execution')
SEED = 20261124
PILOT = ROOT.parent / 'visualmimic_r2_20261124'
COHORT_SIZE = 100


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def checked(ref):
    data = Path(ref['uri']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref['sha256'], ref['uri']
    return data


def load(ref):
    return json.loads(checked(ref))


def ref(path):
    return {'uri': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'media_type': 'video/mp4', 'schema_version': '1'}


def rows():
    return json.loads((OUT / 'inputs.json').read_text())


def arrays(reference):
    import numpy as np
    with np.load(io.BytesIO(checked(reference)), allow_pickle=False) as z:
        return dict(z)


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Protocol already frozen'
    cohort_path = ROOT / 'campaigns/motionfix_target_semantic_v1/tasks.json'
    cohort = json.loads(cohort_path.read_text())[:100]
    previous = json.loads((ROOT / 'campaigns/luna_extend100_v1/combined_report.json').read_text())
    assert [r['key'] for r in cohort] == [r['key'] for r in previous['items']]
    positions = list(range(COHORT_SIZE))
    frozen_settings = json.loads((OUT / 'matched_settings.json').read_text())
    settings = {r['key']['sample_id']: r for r in frozen_settings['items']}
    selected = []
    for i in positions:
        original = cohort[i]
        item = {'key': original['key'], 'instruction': original['instruction'],
                'cohort_position': i, 'source_video': original['source_video']}
        checked(item['source_video']['video'])
        setting = settings[original['key']['sample_id']]
        assert setting['A']['settings_sha256'] == setting['B']['settings_sha256']
        item['matched_settings_sha256'] = setting['A']['settings_sha256']
        for branch in ('A', 'B'):
            result_ref = original[branch]['result']
            assert result_ref['sha256'] == setting[branch]['execution_result_sha256']
            result = load(result_ref)
            retarget = load(result['retarget_report'])
            assert result['key'] == retarget['key'] == item['key']
            assert result['instruction'] == item['instruction']
            assert result['rollout']['outcome'] == 'completed'
            assert retarget['human']['sha256'] == setting[branch]['edited_motion_sha256']
            for artifact in (retarget['human'], result['reference']['motion_npz'], result['rollout']['trajectory']):
                checked(artifact)
            item[branch] = {'result': result_ref, 'human': retarget['human'],
                            'reference': result['reference']['motion_npz'],
                            'trajectory': result['rollout']['trajectory'], 'robot': result['robot']}
        selected.append(item)
    original_protocol = json.loads((ROOT / 'campaigns/luna_extend100_v1/shards/0/protocol.json').read_text())
    protocol = {
        'purpose': 'Locate A/B instruction-success differences at editing, retargeting and execution stages',
        'population': 'The previously judged 100 cases, all with completed A/B rollouts',
        'selection': 'All 100 cases in the original judging cohort; full-cohort endpoint fixed before additional scoring; extension follows the inspected 20-case pilot',
        'seed': SEED, 'positions': positions, 'sample_ids': [r['key']['sample_id'] for r in selected],
        'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'conditions': CONDITIONS, 'requests': 600, 'new_requests': 480, 'judge': 'gpt-6-luna',
        'prompt': original_protocol['prompt'], 'reasoning': 'medium', 'max_output_tokens': 2048,
        'sampling': 'Nearest actual frame at 8 FPS plus endpoints; timestamp labels; no time stretching',
        'human_render': 'A/B share one fixed camera per case; original orientation and temporal motion preserved; constant initial XY centering and constant per-clip floor shift only',
        'robot_render': 'All four reference/execution conditions share one fixed camera per case; constant initial XY centering only; recorded poses, no physics or policy replay',
        'source': 'Same original source video for every condition of a case',
        'judge_blinding': 'Payloads contain SOURCE, CANDIDATE and instruction only; no stage, branch, historical verdict or case identifier',
        'outcomes': ['success', 'failure', 'uncertain', 'technical_error'],
        'analysis': 'Fixed denominator 100; A-only/B-only successes, paired bootstrap confidence intervals and exact paired binary test; three stage comparisons are exploratory',
        'limits': 'Selected completed-rollout cohort VLM study; no human validation, no causal localization claim; A/B input intervention combines reconstruction with downstream editing',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    retain_pilot(selected, protocol)
    save(OUT / 'inputs.json', selected)
    save(OUT / 'protocol.json', protocol)
    print('FROZEN', json.dumps({'cases': COHORT_SIZE, 'sample_ids': protocol['sample_ids'], 'retained_requests': 120, 'new_requests': 480, 'requests': 600}), flush=True)



def retain_pilot(selected, protocol):
    prior_protocol = json.loads((PILOT / 'protocol.json').read_text())
    prior_inputs = json.loads((PILOT / 'inputs.json').read_text())
    prior_results = json.loads((PILOT / 'results.json').read_text())
    ids = set(prior_protocol['sample_ids'])
    assert len(ids) == 20 and len(prior_results) == 120
    assert [r for r in selected if r['key']['sample_id'] in ids] == prior_inputs
    fields = ('conditions', 'judge', 'prompt', 'reasoning', 'max_output_tokens',
              'sampling', 'human_render', 'robot_render', 'source', 'judge_blinding', 'outcomes')
    for field in fields:
        assert protocol[field] == prior_protocol[field] or tuple(protocol[field]) == tuple(prior_protocol[field]), field
    assert prior_protocol['script_sha256'] == hashlib.sha256((PILOT / 'experiment.py').read_bytes()).hexdigest()
    assert {(r['key']['sample_id'], r['condition']) for r in prior_results} == {(sid, c) for sid in ids for c in CONDITIONS}
    assert all(r['status'] == 'assessed' for r in prior_results)
    snapshot = []
    paths = [PILOT / n for n in ['experiment.py', 'protocol.json', 'inputs.json', 'results.json', 'report.json', 'verification.json']]
    for sid in sorted(ids):
        paths.extend(p for p in (PILOT / 'videos' / sid).rglob('*') if p.is_file())
        for condition in CONDITIONS:
            paths.extend(p for p in (PILOT / 'judgments' / (sid + '_' + condition)).rglob('*') if p.is_file())
    for path in sorted(set(paths)):
        snapshot.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save(OUT / 'retained_pilot_manifest.json', snapshot)
    for directory in ('videos', 'judgments'):
        (OUT / directory).mkdir(exist_ok=False)
    for sid in sorted(ids):
        (OUT / 'videos' / sid).symlink_to(PILOT / 'videos' / sid, target_is_directory=True)
        for condition in CONDITIONS:
            name = sid + '_' + condition
            (OUT / 'judgments' / name).symlink_to(PILOT / 'judgments' / name, target_is_directory=True)
    protocol['extension'] = {
        'pilot': str(PILOT), 'retained_sample_ids': sorted(ids), 'new_cases': 80,
        'pilot_protocol_sha256': hashlib.sha256((PILOT / 'protocol.json').read_bytes()).hexdigest(),
        'pilot_results_sha256': hashlib.sha256((PILOT / 'results.json').read_bytes()).hexdigest(),
        'pilot_file_manifest_sha256': hashlib.sha256((OUT / 'retained_pilot_manifest.json').read_bytes()).hexdigest(),
        'reuse': 'All 120 pilot verdicts and their exact videos, requests and responses are reused without modification or reselection',
        'scope': 'No new human ratings, no new policy execution, no repository or document edits',
    }


def retained_render(row, conditions):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    sid = row['key']['sample_id']
    if sid not in protocol['extension']['retained_sample_ids']:
        return False
    for condition in conditions:
        metadata = json.loads((OUT / 'videos' / sid / (condition + '.json')).read_text())
        checked(metadata['video'])
    print('REUSED_RENDER', sid, ','.join(conditions), flush=True)
    return True


def human():
    import numpy as np
    import torch
    import smplx
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    from moderngl_window.context.headless.window import Window
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT))
    from visualmimic.contracts import ArtifactRef
    from visualmimic.storage import ArtifactStore
    store = ArtifactStore(OUT / 'reader', [ROOT])
    model = smplx.SMPLHLayer(str(ROOT / 'visualmimic/assets/body_models/smplh'), gender='neutral', ext='npz').eval()
    CONFIG.update_conf({'z_up': False, 'playback_fps': 30, 'auto_set_floor': False, 'auto_set_camera_target': False})

    class EGLWindow(Window):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, backend='egl')

    for row in rows():
        if retained_render(row, ('A_human', 'B_human')):
            continue
        sid = row['key']['sample_id']
        vertices, timestamps, shifts = {}, {}, {}
        for branch in ('A', 'B'):
            motion = store.read_motion(ArtifactRef(**row[branch]['human']))
            ts = motion.timestamps_s
            assert np.allclose(np.diff(ts), 1 / 30, atol=1e-6)
            rots = Rotation.from_rotvec(motion.fields['rots'].reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
            trans = motion.fields['trans'].copy()
            xy = trans[0, :2].copy()
            trans[:, :2] -= xy
            with torch.inference_mode():
                mesh = model(global_orient=torch.from_numpy(rots[:, :1]).float(), body_pose=torch.from_numpy(rots[:, 1:]).float(), transl=torch.from_numpy(trans).float())
            v = mesh.vertices.numpy()[..., [0, 2, 1]].copy()
            v[..., 2] *= -1
            floor_shift = -float(v[..., 1].min())
            v[..., 1] += floor_shift
            vertices[branch] = v
            timestamps[branch] = ts.tolist()
            shifts[branch] = {'initial_xy_translation': xy.tolist(), 'constant_floor_shift_m': floor_shift}
        low = np.minimum(vertices['A'].min((0, 1)), vertices['B'].min((0, 1)))
        high = np.maximum(vertices['A'].max((0, 1)), vertices['B'].max((0, 1)))
        target = (low + high) / 2
        distance = max(3., float(np.linalg.norm(high - low) / 2 / np.sin(np.deg2rad(22.5)) * 1.25))
        direction = np.array([1., .35, 1.]); direction /= np.linalg.norm(direction)
        position = target + direction * distance
        for branch in ('A', 'B'):
            condition = branch + '_human'
            path = OUT / 'videos' / sid / (condition + '.mp4')
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists()
            with patch('aitviewer.viewer.get_local_window_cls', return_value=EGLWindow):
                viewer = HeadlessRenderer(size=(640, 480))
            try:
                viewer.auto_set_floor = False; viewer.auto_set_camera_target = False
                viewer.scene.add(Meshes(vertices[branch], model.faces, color=(.65, .65, .65, 1), z_up=False))
                viewer.scene.camera.position = position
                viewer.scene.camera.target = target
                viewer.scene.background_color = np.array([1., 1., 1., 1.])
                with imageio.get_writer(str(path), fps=30, codec='libx264', quality=8, macro_block_size=1) as writer:
                    for i in range(len(vertices[branch])):
                        viewer.scene.current_frame_id = i
                        frame = np.asarray(viewer.get_frame())
                        assert frame.shape[:2] == (480, 640)
                        writer.append_data(frame[:, :, :3])
            finally:
                viewer.on_close()
            save(path.with_suffix('.json'), {'video': ref(path), 'timestamps_s': timestamps[branch],
                'source_motion': row[branch]['human'], 'camera_position_yup': position.tolist(),
                'camera_target_yup': target.tolist(), 'transforms': shifts[branch], 'fps': 30})
        print('HUMAN_RENDERED', sid, flush=True)


def robot():
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    import mjlab
    xml = Path(mjlab.__file__).parent / 'asset_zoo/robots/unitree_g1/xmls/g1.xml'
    xml_sha = hashlib.sha256(xml.read_bytes()).hexdigest()
    for row in rows():
        if retained_render(row, ('A_reference', 'B_reference', 'A_execution', 'B_execution')):
            continue
        sid = row['key']['sample_id']
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.worldbody.add_geom(name='video_floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1], rgba=[.22, .25, .30, 1])
        spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])
        model = spec.compile(); model.vis.global_.offwidth = 640; model.vis.global_.offheight = 480
        data = mujoco.MjData(model)
        states, times, shifts, fk_errors = {}, {}, {}, {}
        bounds = []
        for branch in ('A', 'B'):
            assert xml_sha == row[branch]['robot']['model_asset']['sha256']
            a = arrays(row[branch]['reference']); fps = float(a['fps'].reshape(-1)[0])
            assert a['body_names'][0] == 'pelvis'
            qref = np.concatenate([a['body_pos_w'][:, 0], a['body_quat_w'][:, 0], a['joint_pos']], axis=1)
            rollout = arrays(row[branch]['trajectory'])
            for stage, q, ts in [('reference', qref, np.arange(len(qref)) / fps),
                                 ('execution', rollout['states'][:, :36], rollout['timestamps_s'])]:
                name = branch + '_' + stage
                stride = round(1 / (25 * float(ts[1] - ts[0])))
                assert stride >= 1 and np.isclose(stride * float(ts[1] - ts[0]), 1 / 25)
                ids = list(range(0, len(q), stride))
                if ids[-1] != len(q) - 1: ids.append(len(q) - 1)
                q = q[ids].copy(); shift = q[0, :2].copy(); q[:, :2] -= shift
                states[name] = q; times[name] = ts[ids].tolist(); shifts[name] = shift.tolist()
                err = 0.
                for j, pose in enumerate(q):
                    data.qpos[:] = pose; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                    bounds.append(data.xpos[1:].copy())
                    if stage == 'reference':
                        expected = a['body_pos_w'][ids[j]].copy(); expected[:, :2] -= shift
                        err = max(err, float(np.max(np.abs(data.xpos[1:] - expected))))
                if stage == 'reference': assert err < 1e-4, (sid, name, err)
                fk_errors[name] = err if stage == 'reference' else None
        positions = np.asarray(bounds)
        low, high = positions.min((0, 1)) - .25, positions.max((0, 1)) + .25
        target = (low + high) / 2
        distance = max(3.4, float(np.linalg.norm(high - low) / 2 / np.sin(np.deg2rad(22.5)) * 1.25))
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 135.; camera.elevation = -18.; camera.distance = distance; camera.lookat[:] = target
        option = mujoco.MjvOption(); option.geomgroup[3:] = False
        renderer = mujoco.Renderer(model, height=480, width=640)
        try:
            for name, q in states.items():
                path = OUT / 'videos' / sid / (name + '.mp4')
                path.parent.mkdir(parents=True, exist_ok=True); assert not path.exists()
                with imageio.get_writer(str(path), fps=25, codec='libx264', quality=8) as writer:
                    for pose in q:
                        data.qpos[:] = pose; data.qvel[:] = 0; mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        writer.append_data(renderer.render().copy())
                branch = name[0]; artifact = 'reference' if name.endswith('reference') else 'trajectory'
                save(path.with_suffix('.json'), {'video': ref(path), 'timestamps_s': times[name],
                    'source_artifact': row[branch][artifact], 'initial_xy_translation': shifts[name],
                    'camera': {'azimuth': 135., 'elevation': -18., 'distance': distance, 'lookat': target.tolist(), 'mode': 'fixed'},
                    'fps': 25, 'model_sha256': xml_sha, 'max_fk_error_m': fk_errors[name], 'new_physics_execution': False})
        finally:
            renderer.close()
        print('ROBOT_RENDERED', sid, flush=True)


def frames(video, label):
    import numpy as np
    import cv2
    checked(video['video'])
    ts = video['timestamps_s']; wanted = np.arange(ts[0], ts[-1] + 1e-9, .125)
    ids = sorted(set([int(np.abs(np.asarray(ts) - t).argmin()) for t in wanted] + [0, len(ts) - 1]))
    capture = cv2.VideoCapture(video['video']['uri'])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(ts)
    content = [{'type': 'input_text', 'text': f'{label}: duration {ts[-1] - ts[0]:.3f}s. Chronological frames follow.'}]
    try:
        for i in ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = capture.read(); assert ok
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85]); assert ok
            content += [{'type': 'input_text', 'text': f'{label} t={ts[i]:.3f}s'},
                        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode(), 'detail': 'high'}]
    finally:
        capture.release()
    return content, ids


def judge():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    credentials = json.loads((ROOT / 'judge-secrets/luna.json').read_text())
    assert credentials['JUDGE_BASE_URL'].rstrip('/') == 'https://api.openai.com/v1'
    assert credentials['JUDGE_MODEL'] == protocol['judge']
    tasks = [(row, name) for row in rows() for name in CONDITIONS]
    random.Random(SEED).shuffle(tasks)
    schema = {'type': 'object', 'properties': {'verdict': {'type': 'string', 'enum': ['success', 'failure', 'uncertain']},
              'reason': {'type': 'string'}}, 'required': ['verdict', 'reason'], 'additionalProperties': False}

    def evaluate(task):
        row, name = task; sid = row['key']['sample_id']
        destination = OUT / 'judgments' / (sid + '_' + name)
        if (destination / 'result.json').exists():
            return json.loads((destination / 'result.json').read_text())
        destination.mkdir(parents=True, exist_ok=False)
        candidate = json.loads((OUT / 'videos' / sid / (name + '.json')).read_text())
        source_content, source_ids = frames(row['source_video'], 'SOURCE')
        candidate_content, candidate_ids = frames(candidate, 'CANDIDATE')
        payload = {'model': protocol['judge'], 'store': False, 'reasoning': {'effort': 'medium'},
                   'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': protocol['prompt'] + '\nINSTRUCTION: ' + row['instruction']}] + source_content + candidate_content}],
                   'text': {'format': {'type': 'json_schema', 'name': 'motion_edit_success', 'strict': True, 'schema': schema}}, 'max_output_tokens': 2048}
        body = json.dumps(payload).encode()
        (destination / 'request.json.gz').write_bytes(gzip.compress(body, mtime=0))
        result = {'key': row['key'], 'instruction': row['instruction'], 'condition': name,
                  'source_video_sha256': row['source_video']['video']['sha256'], 'candidate_video_sha256': candidate['video']['sha256'],
                  'source_frame_indices': source_ids, 'candidate_frame_indices': candidate_ids,
                  'request_sha256': hashlib.sha256(body).hexdigest(), 'status': 'inflight'}
        save(destination / 'request_metadata.json', result)
        started = time.time()
        try:
            request = urllib.request.Request(credentials['JUDGE_BASE_URL'].rstrip('/') + '/responses', data=body,
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + credentials['JUDGE_API_KEY']})
            with urllib.request.urlopen(request, timeout=240) as response:
                raw_bytes = response.read()
            (destination / 'response.json').write_bytes(raw_bytes)
            raw = json.loads(raw_bytes)
            assert raw.get('status') == 'completed', 'Incomplete response'
            text = ''.join(c['text'] for item in raw.get('output', []) for c in item.get('content', []) if c.get('type') == 'output_text')
            answer = json.loads(text)
            assert answer['verdict'] in ('success', 'failure', 'uncertain')
            result.update(status='assessed', verdict=answer['verdict'], reason=answer['reason'], response_model=raw.get('model'),
                          response_sha256=hashlib.sha256(raw_bytes).hexdigest(), output_text=text, usage=raw.get('usage'))
        except urllib.error.HTTPError as error:
            result.update(status='technical_error', verdict='technical_error', error='HTTP ' + str(error.code))
        except Exception as error:
            result.update(status='technical_error', verdict='technical_error', error=type(error).__name__ + ': ' + str(error)[:300])
        result['seconds'] = time.time() - started
        save(destination / 'result.json', result)
        return result

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); completed.append(result)
            save(OUT / 'progress.json', {'finished': len(completed), 'expected': len(tasks),
                 'technical_errors': sum(r['verdict'] == 'technical_error' for r in completed)})
            print('JUDGED', len(completed), result['key']['sample_id'], result['condition'], result['verdict'], flush=True)
    save(OUT / 'results.json', sorted(completed, key=lambda r: (r['key']['sample_id'], r['condition'])))


def summarize():
    from collections import Counter
    data = json.loads((OUT / 'results.json').read_text())
    assert len(data) == len({(r['key']['sample_id'], r['condition']) for r in data}) == COHORT_SIZE * len(CONDITIONS)
    index = {(r['key']['sample_id'], r['condition']): r for r in data}
    ids = [r['key']['sample_id'] for r in rows()]
    groups = {c: dict(Counter(r['verdict'] for r in data if r['condition'] == c)) for c in CONDITIONS}
    paired = {}
    for stage in ('human', 'reference', 'execution'):
        pairs = [(index[(sid, 'A_' + stage)]['verdict'], index[(sid, 'B_' + stage)]['verdict']) for sid in ids]
        a_only = sum(a == 'success' and b != 'success' for a, b in pairs)
        b_only = sum(b == 'success' and a != 'success' for a, b in pairs)
        d = [(b == 'success') - (a == 'success') for a, b in pairs]
        rng = random.Random(SEED)
        bootstrap = sorted(sum(rng.choices(d, k=len(d))) / len(d) * 100 for _ in range(10000))
        discordant = a_only + b_only
        exact_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2 ** discordant) if discordant else 1.
        paired[stage] = {'n': len(pairs), 'A_only_success': a_only, 'B_only_success': b_only,
            'both_success': sum(a == b == 'success' for a, b in pairs), 'neither_success': sum(a != 'success' and b != 'success' for a, b in pairs),
            'B_minus_A_percentage_points': sum(d) / len(d) * 100,
            'paired_bootstrap_95_percent_interval': [bootstrap[249], bootstrap[9749]],
            'exact_paired_binary_p': exact_p,
            'unresolved_pairs': sum(a in ('uncertain', 'technical_error') or b in ('uncertain', 'technical_error') for a, b in pairs)}
    transitions = {}
    for branch in ('A', 'B'):
        for before, after in [('human', 'reference'), ('reference', 'execution')]:
            transitions[branch + '_' + before + '_to_' + after] = dict(Counter(index[(sid, branch + '_' + before)]['verdict'] + ' -> ' + index[(sid, branch + '_' + after)]['verdict'] for sid in ids))
    report = {'cases': len(ids), 'judgments': len(data), 'groups': groups, 'paired': paired, 'transitions': transitions,
              'response_models': sorted({r.get('response_model', 'error') for r in data}),
              'technical_errors': sum(r['verdict'] == 'technical_error' for r in data),
              'note': 'Success versus all other outcomes on the fixed cohort; uncertainty and errors also reported separately. Exploratory intervals are not adjusted for three stage comparisons.'}
    save(OUT / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'human', 'robot', 'judge', 'summarize'])
    globals()[parser.parse_args().stage]()
