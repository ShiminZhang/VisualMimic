from dataclasses import asdict
import json
import numpy as np
import pytest
from visualmimic.batch_edit import run_batch
from visualmimic.contracts import BackendIdentity, GenerationOptions, MotionSequence
from visualmimic.human_motion import HUMAN_SPEC
from visualmimic.storage import ArtifactStore, digest


class FixtureEditor:
    identity = BackendIdentity('fixture','1','fixture',fixture=True)
    def __init__(self): self.calls=0; self.interrupt=False
    def edit(self,source,instruction,options):
        self.calls+=1
        if self.interrupt: raise KeyboardInterrupt()
        return source


def inputs(tmp_path):
    store=ArtifactStore(tmp_path/'artifacts')
    motion=MotionSequence(HUMAN_SPEC,np.arange(2)/30.,{'rots':np.zeros((2,22,3),np.float32),
        'trans':np.zeros((2,3),np.float32),'joint_positions':np.zeros((2,22,3),np.float32)},np.ones(2,bool))
    rows=[{'key':{'dataset_version':'fixture','sample_id':'one','split':'test'},
           'instruction':'test','source':asdict(store.put_motion(motion))}]
    return store,rows


def test_resume_verifies_completed_artifact_without_replaying(tmp_path):
    store,rows=inputs(tmp_path);editor=FixtureEditor();opt=GenerationOptions(1,'fixture','fixture')
    run_batch(rows,store,editor,opt,tmp_path/'run')
    run_batch(rows,store,editor,opt,tmp_path/'run',resume=True)
    assert editor.calls == 1
    with pytest.raises(ValueError,match='mismatch'):
        run_batch(rows,store,editor,GenerationOptions(2,'fixture','fixture'),tmp_path/'run',resume=True)


def test_interruption_requires_explicit_retry(tmp_path):
    store,rows=inputs(tmp_path);editor=FixtureEditor();editor.interrupt=True;opt=GenerationOptions(1,'fixture','fixture')
    with pytest.raises(KeyboardInterrupt):run_batch(rows,store,editor,opt,tmp_path/'run')
    editor.interrupt=False
    with pytest.raises(RuntimeError,match='Uncertain'):
        run_batch(rows,store,editor,opt,tmp_path/'run',resume=True)
    assert editor.calls == 1
    run_batch(rows,store,editor,opt,tmp_path/'run',resume=True,retry_interrupted=True)
    assert editor.calls == 2
    assert json.loads((tmp_path/'run/status.json').read_text())['status']=='completed'
