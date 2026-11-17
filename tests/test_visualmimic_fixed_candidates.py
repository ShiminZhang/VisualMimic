"""Partial runs must not become deceptively easier retrieval benchmarks."""
import json
import subprocess
import sys


def test_missing_prediction_preserves_denominator_and_withholds_metrics(tmp_path):
    prepared = tmp_path/'prepared'
    (prepared/'source').mkdir(parents=True)
    (prepared/'labels').mkdir()
    keys = [{'dataset_version':'test','sample_id':s,'split':'test'} for s in ('a','b')]
    ref = {'uri':'/nonexistent','sha256':'not-read','media_type':'application/json'}
    (prepared/'source/manifest.json').write_text('[]')
    (prepared/'labels/manifest.json').write_text(json.dumps([{'key':k,'target':ref,'source':ref} for k in keys]))
    (prepared/'preparation.json').write_text(json.dumps({'ordered_ids':['a','b'],'hashes':{},'cross_split_segment_count':0}))
    predictions = tmp_path/'predictions.json'
    predictions.write_text(json.dumps([{'key':keys[0],'motion':ref}]))
    out = tmp_path/'report'
    subprocess.run([sys.executable,'-m','visualmimic.evaluate_human','--prepared',str(prepared),
        '--predictions',str(predictions),'--prediction-artifacts',str(tmp_path),
        '--repository','/missing','--tmr-assets','/missing','--output',str(out)],check=True,capture_output=True)
    report = json.loads((out/'report.json').read_text())
    assert report['expected'] == 2 and report['available'] == 1
    assert report['coverage'] == .5
    assert report['metrics'] is None
    assert report['missing'] == [keys[1]]
    assert report['candidate_keys'] == keys
    assert report['status'] == 'incomplete_predictions'
