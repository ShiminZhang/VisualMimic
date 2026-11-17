"""Add the exact pre-execution A reference as C to a frozen G1 retrieval run.

Never modifies the parent campaign. Uses its fixed 307 candidates, targets,
features, metric, tie handling and query bootstrap. No policy replay.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def freeze(parent, output):
    output.mkdir(exist_ok=False)
    names = ['evaluate.py','protocol.json','inputs.json','features.npz',
             'distance_matrices.npz','report.json','provenance.json']
    snapshot = output/'parent_snapshot'
    snapshot.mkdir()
    hashes = {}
    for name in names:
        shutil.copy2(parent/name, snapshot/name)
        hashes[name] = sha(snapshot/name)
        assert hashes[name] == sha(parent/name)
    report = json.loads((snapshot/'report.json').read_text())
    assert report['status'] == 'evaluated' and report['evaluated'] == 307
    assert hashes['evaluate.py'] == report['protocol']['evaluator_sha256']
    save(output/'protocol.json', {
        'scope': 'G1 geometric retrieval, A/B executed vs C exact compiled A reference',
        'parent': str(parent), 'parent_sha256': hashes,
        'expected': 307, 'candidate_sha256': report['protocol']['candidate_sha256'],
        'metric': report['protocol'],
        'C': 'reference.motion_npz from each saved A execution result; no resampling of editor',
        'query': 'same saved GMR target features as parent',
        'missing': 'withhold metrics rather than shrink candidate set',
        'comparisons': ['B_minus_A','A_minus_C','B_minus_C'],
        'bootstrap': '10000 paired query resamples, seed42, candidates fixed',
        'boundary': 'not official TMR, not binary instruction success'})


def load_module(path):
    spec = importlib.util.spec_from_file_location('frozen_g1_retrieval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(output):
    import numpy as np
    protocol = json.loads((output/'protocol.json').read_text())
    snapshot = output/'parent_snapshot'
    for name, expected in protocol['parent_sha256'].items():
        assert sha(snapshot/name) == expected, 'Frozen parent modified'
    old = load_module(snapshot/'evaluate.py')
    old.selftest()
    rows = json.loads((snapshot/'inputs.json').read_text())
    assert len(rows) == 307 and len({old.key(r) for r in rows}) == 307
    assert [r['key'] for r in rows] == protocol['metric']['ordered_keys']
    cache = dict(np.load(snapshot/'features.npz', allow_pickle=False))
    old_dist = dict(np.load(snapshot/'distance_matrices.npz', allow_pickle=False))
    old_report = json.loads((snapshot/'report.json').read_text())
    fields = ('pose','root','velocity','duration')
    features = {g: [{f:cache[g+'_'+f][i] for f in fields} for i in range(307)]
                for g in ('target','A','B')}
    features['C'] = []
    provenance = []
    for i, row in enumerate(rows):
        ref = row['A']['result']
        r = json.loads(old.bytes_ref(ref))
        assert old.key(r) == old.key(row) and r['instruction'] == row['instruction']
        assert r['rollout']['outcome'] == 'completed'
        assert r['rollout']['policy_id'] == old_report['policy_sha256']
        assert r['rollout']['protocol_id'] == old_report['execution_protocol']
        assert r['reference']['reference_id'] == r['rollout']['reference_id'] == r['reference']['motion_npz']['sha256']
        a = old.npz(r['reference']['motion_npz'])
        names = a['body_names'].tolist()
        pelvis = names.index('pelvis')
        indices = [names.index(n) for n in old.BODY_NAMES]
        t = np.arange(len(a['joint_pos']))/float(np.asarray(a['fps']).reshape(-1)[0])
        assert len(t) == r['reference']['num_frames']
        assert np.isclose(t[-1],r['reference']['duration_s'])
        points = a['body_pos_w'][:,indices,:].astype(float)
        root = a['body_pos_w'][:,pelvis,:].astype(float)
        quat = a['body_quat_w'][:,pelvis,:].astype(float)
        assert np.isfinite(points).all() and np.isfinite(root).all() and np.isfinite(quat).all()
        features['C'].append(old.featurize(points,root,quat,t))
        provenance.append({'key':row['key'],'A_result':ref,'C_reference':r['reference']['motion_npz']})
        if i%25 == 0: print('C_FEATURES',i+1,'/307',flush=True)
    # Check compiled FK against independent MuJoCo FK on a representative clip.
    import mujoco
    from supermimic.data.motion_convert import load_g1_model,g1_xml_path
    assert sha(g1_xml_path()) == r['robot']['model_asset']['sha256']
    model=load_g1_model(); data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in old.BODY_NAMES]
    assert min(ids)>=0
    for frame in (0,len(t)//2,len(t)-1):
        data.qpos[:]=np.concatenate([root[frame],quat[frame],a['joint_pos'][frame]])
        mujoco.mj_forward(model,data)
        assert np.allclose(data.xpos[ids],points[frame],atol=2e-6)
    rng=np.random.default_rng(42)
    boots=rng.integers(0,307,(10000,307))
    results={};matrices={}
    for mode in ('composite','pose_only'):
        ds={g:old.matrix(features['target'],features[g],mode) for g in ('A','B','C')}
        rr={g:old.ranks(ds[g]) for g in ds}
        for g in ('A','B'):
            assert np.allclose(ds[g],old_dist[mode+'_'+g],atol=1e-9,rtol=1e-8)
            assert old.metrics(rr[g]) == old_report['results'][mode][g]
        comparisons={}
        for later,baseline in [('B','A'),('A','C'),('B','C')]:
            changes={}
            for metric in ('R@1','R@5','MedR'):
                if metric.startswith('R@'):
                    k=int(metric[2:]);delta=100*((rr[later]<=k).astype(float)-(rr[baseline]<=k).astype(float))
                    estimate=delta.mean();samples=delta[boots].mean(1)
                else:
                    estimate=np.median(rr[later])-np.median(rr[baseline])
                    samples=np.median(rr[later][boots],axis=1)-np.median(rr[baseline][boots],axis=1)
                changes[metric]={'difference':float(estimate),'paired_query_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist()}
            delta=np.diag(ds[later])-np.diag(ds[baseline])
            changes['matched_distance']={'mean_difference':float(delta.mean()),
                 'paired_query_bootstrap_95ci':np.quantile(delta[boots].mean(1),[.025,.975]).tolist()}
            comparisons[later+'_minus_'+baseline]=changes
        results[mode]={'metrics':{g:old.metrics(rr[g]) for g in rr},
                       'matched_distance_mean':{g:float(np.diag(ds[g]).mean()) for g in ds},
                       'paired_changes':comparisons,
                       'items':[{'key':row['key'],'ranks':{g:float(rr[g][i]) for g in rr},
                                 'matched_distance':{g:float(ds[g][i,i]) for g in ds}} for i,row in enumerate(rows)]}
        matrices.update({mode+'_'+g:d for g,d in ds.items()})
    np.savez_compressed(output/'distance_matrices.npz',**matrices)
    np.savez_compressed(output/'C_features.npz',**{f:np.stack([r[f] for r in features['C']]) for f in fields})
    save(output/'provenance.json',provenance)
    save(output/'report.json',{'status':'evaluated','expected':307,'evaluated':307,
        'protocol':protocol,'results':results,'validation':{'parent_AB_reproduced':True,'C_FK_checked':True},
        'limitations':['selected both-completed cohort','geometric proxy, not original TMR or success rate',
                       'initial heading removed; phase normalization retains velocity/duration only through composite',
                       'fixed parent weights; no tuning for C; target retargeting still introduces error']})
    print(json.dumps({mode:{k:v for k,v in x.items() if k!='items'} for mode,x in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);p.add_argument('--parent',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a.parent,a.output)
    else:
        save(a.output/'status.json',{'status':'running'})
        try:
            run(a.output);save(a.output/'status.json',{'status':'evaluated'})
        except Exception as exc:
            save(a.output/'status.json',{'status':'failed','error':str(exc)});raise
