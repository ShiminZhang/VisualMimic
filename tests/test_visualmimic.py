import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np
from visualmimic.contracts import *
from visualmimic.storage import ArtifactStore, digest
from visualmimic.adaptation import IdentityAdapter
from visualmimic.dataset import DatasetRecord, prepare, video_requests, source_requests
from visualmimic.reconstruction import ReconstructionResult
from visualmimic.runner import Runner, RunConfig
from visualmimic.evaluation import EvaluationProtocol, evaluate, compare_reports
from visualmimic.diagnostics import compare_reconstruction


class FixtureEditor:
    identity = BackendIdentity('test-editor', '1', 'fixture-config', fixture=True)

    def __init__(self, spec):
        self.input_spec = self.output_spec = spec
        self.calls = []

    def edit(self, source, instruction, options):
        self.calls.append((instruction, options))
        return MotionSequence(source.spec, source.timestamps_s.copy(),
                              {k: v + np.float32(1) for k, v in source.fields.items()}, source.valid_mask.copy())


class FixtureReconstructor:
    identity = BackendIdentity('test-reconstruction', '1', 'fixture-config', fixture=True)

    def __init__(self, store, motion, fail=False):
        self.store, self.motion, self.fail = store, motion, fail

    def reconstruct(self, video):
        if self.fail:
            raise RuntimeError('Deliberate reconstruction failure')
        return ReconstructionResult(self.store.put_motion(self.motion), None, self.store.put_json({'fixture': True}))


class FixtureMetrics:
    identity = BackendIdentity('test-metrics', '1', 'fixture-config', fixture=True)

    def evaluate(self, predictions, references, protocol):
        return {'fixture_count': len(predictions)}


class VisualMimicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(Path(self.tmp.name) / 'inputs')
        self.labels = ArtifactStore(Path(self.tmp.name) / 'labels', [self.store.root])
        self.spec = MotionSpec('test-v1', 'positions', 'test-skeleton', 'right-handed-z-up', 'metres',
                               {'positions': FieldSpec((2, 3))})
        self.motion = MotionSequence(self.spec, np.array([0., .1]),
                                     {'positions': np.zeros((2, 2, 3), dtype=np.float32)}, np.array([True, True]))
        self.key = SampleKey('test-v1', 'sample', 'test')
        self.source = self.store.put_motion(self.motion)
        self.request = Source3DRequest(self.key, 'fixture instruction', self.source)
        self.video = VideoInput(self.key, self.store.put_bytes(b'fixture-video', 'video/test'), (0., .1), 16, 16)
        self.video_request = VideoRequest(self.key, self.request.instruction, self.video)
        self.editor = FixtureEditor(self.spec)
        self.reconstructor = FixtureReconstructor(self.store, self.motion)
        self.runner = Runner(self.store, IdentityAdapter(self.store), self.editor, self.reconstructor)
        self.config = RunConfig(GenerationOptions(42, 'source-duration', 'fixture-v1'), allow_fixture=True)

    def protocol(self, keys=None):
        return EvaluationProtocol(FixtureMetrics.identity, self.store.put_bytes(b'weights'),
                                  self.store.put_json(['ordered', 'candidates']), 'none', 'one-sample',
                                  tuple(keys or [self.key]), 42)

    def refs(self):
        return [EvalReference(self.key, self.source, self.source)]

    def test_roundtrip_and_tamper_detection(self):
        loaded = self.store.read_motion(self.source)
        np.testing.assert_array_equal(loaded.fields['positions'], self.motion.fields['positions'])
        Path(self.source.uri).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.store.read_motion(self.source)

    def test_reject_bad_shape_timestamps_and_nonfinite(self):
        for motion in [dataclasses.replace(self.motion, timestamps_s=np.array([0., 0.])),
                       dataclasses.replace(self.motion, fields={'positions': np.zeros((2, 3), dtype=np.float32)}),
                       dataclasses.replace(self.motion, fields={'positions': np.full((2, 2, 3), np.nan, dtype=np.float32)})]:
            with self.assertRaises(ValueError):
                motion.validate()

    def test_store_blocks_label_reads_and_symlink_escape(self):
        secret = self.labels.put_json({'target': 'private'})
        with self.assertRaises(PermissionError):
            self.store.read_json(secret)
        link = self.store.root / 'escape'
        link.symlink_to(secret.uri)
        with self.assertRaises(PermissionError):
            self.store.read_json(dataclasses.replace(secret, uri=str(link)))

    def test_video_manifest_rejects_extra_label_fields(self):
        row = dataclasses.asdict(self.video_request)
        row['target'] = dataclasses.asdict(self.source)
        manifest = self.store.put_json([row])
        with self.assertRaises(ValueError):
            list(video_requests(self.store, manifest, 'test'))

    def test_dataset_views_and_cross_split_detection(self):
        r = DatasetRecord(self.key, 'edit', 'src', 'dst', self.source, self.source)
        prepared = prepare([r], self.store, self.labels)
        self.assertEqual(len(list(source_requests(self.store, prepared.source_manifest, 'test'))), 1)
        with self.assertRaises(PermissionError):
            self.store.read_json(prepared.label_manifest)
        other = dataclasses.replace(r, key=SampleKey('test-v1', 'other', 'train'))
        with self.assertRaisesRegex(ValueError, 'across'):
            prepare([r, other], self.store, self.labels)

    def test_both_paths_share_editor_and_prediction(self):
        a = self.runner.run_source(self.request, self.config)
        b = self.runner.run_video(self.video_request, self.config)
        self.assertEqual((a.status, b.status), ('succeeded', 'succeeded'))
        self.assertEqual(a.prediction.motion.sha256, b.prediction.motion.sha256)
        self.assertEqual(self.editor.calls[0], self.editor.calls[1])
        ta, tb = self.store.read_json(a.trace), self.store.read_json(b.trace)
        self.assertEqual(ta['comparison_id'], tb['comparison_id'])
        self.assertNotEqual(ta['run_fingerprint'], tb['run_fingerprint'])
        self.assertNotIn('target', tb['request'])
        self.assertNotIn('source', tb['request'])

    def test_fixtures_disabled_by_default(self):
        r = self.runner.run_source(self.request, dataclasses.replace(self.config, allow_fixture=False))
        self.assertEqual(r.error.stage, 'configuration')
        self.assertIsNone(r.prediction)
        self.assertFalse(self.editor.calls)

    def test_reconstruction_failure_is_not_filled_with_source(self):
        self.reconstructor.fail = True
        r = self.runner.run_video(self.video_request, self.config)
        self.assertEqual(r.error.stage, 'reconstruction')
        self.assertIsNone(r.prediction)
        self.assertFalse(self.editor.calls)

    def test_unknown_spec_fails_in_adaptation(self):
        self.editor.input_spec = dataclasses.replace(self.spec, units='centimetres')
        r = self.runner.run_source(self.request, self.config)
        self.assertEqual(r.error.stage, 'adaptation')
        self.assertFalse(self.editor.calls)

    def test_seed_changes_fingerprint_and_comparison(self):
        a = self.runner.run_source(self.request, self.config)
        b = self.runner.run_source(self.request, dataclasses.replace(self.config, options=dataclasses.replace(self.config.options, seed=43)))
        ta, tb = self.store.read_json(a.trace), self.store.read_json(b.trace)
        self.assertNotEqual(ta['run_fingerprint'], tb['run_fingerprint'])
        self.assertNotEqual(ta['comparison_id'], tb['comparison_id'])

    def test_coverage_includes_failed_and_missing_samples(self):
        self.reconstructor.fail = True
        r = self.runner.run_video(self.video_request, self.config)
        missing_key = SampleKey('test-v1', 'missing', 'test')
        refs = self.refs() + [EvalReference(missing_key, self.source, self.source)]
        report = evaluate([r], refs, self.protocol([self.key, missing_key]), FixtureMetrics(), self.store)
        self.assertEqual((report['expected'], report['succeeded'], report['failed'], report['missing']), (2, 0, 1, 1))
        self.assertIsNone(report['metrics'])
        self.assertEqual(report['failure_stages'], {'reconstruction': 1})

    def test_reports_compare_and_reject_changed_settings(self):
        a = self.runner.run_source(self.request, self.config)
        b = self.runner.run_video(self.video_request, self.config)
        reports = [evaluate([r], self.refs(), self.protocol(), FixtureMetrics(), self.store) for r in (a, b)]
        self.assertEqual(compare_reports(*reports)['common_successful_count'], 1)
        with self.assertRaises(ValueError):
            compare_reports(reports[0], dict(reports[1], comparison_id='different'))

    def test_duplicate_or_forged_results_rejected(self):
        r = self.runner.run_source(self.request, self.config)
        for results in ([r, r], [dataclasses.replace(r, seed=43)], [dataclasses.replace(r, prediction=dataclasses.replace(r.prediction, motion=self.source))]):
            with self.assertRaises(ValueError):
                evaluate(results, self.refs(), self.protocol(), FixtureMetrics(), self.store)

    def test_renderer_publishes_only_video_view(self):
        from visualmimic.dataset import render_dataset
        from visualmimic.rendering import RenderConfig, RenderResult
        store, labels = self.store, self.labels
        class FixtureRenderer:
            def render(self, source, config):
                return RenderResult(store.put_bytes(b'video', 'video/test'), (0., .1),
                                    digest(config), labels.put_json({'camera_truth': 'private'}))
        config = RenderConfig('fixed', 'fixed', 'fixed', 'fixed', 10., 16, 16)
        manifest = render_dataset([self.request], store, store, FixtureRenderer(), config)
        rows = list(video_requests(store, manifest, 'test'))
        self.assertEqual(len(rows), 1)
        self.assertNotIn('camera_truth', str(store.read_json(manifest)))
        self.assertNotIn('source', store.read_json(manifest)[0])

    def test_pairing_rejects_changed_instruction(self):
        a = self.runner.run_source(self.request, self.config)
        b = self.runner.run_video(dataclasses.replace(self.video_request, instruction='different'), self.config)
        reports = [evaluate([r], self.refs(), self.protocol(), FixtureMetrics(), self.store) for r in (a, b)]
        with self.assertRaisesRegex(ValueError, 'instructions'):
            compare_reports(*reports)

    def test_diagnostics_require_matching_time_and_spec(self):
        report = compare_reconstruction(self.motion, self.motion)
        self.assertEqual(report['per_field_rmse']['positions'], 0)
        with self.assertRaises(ValueError):
            compare_reconstruction(dataclasses.replace(self.motion, timestamps_s=np.array([0., .2])), self.motion)


if __name__ == '__main__':
    unittest.main()
