"""Label-free stage-one runners. No cache reuse until manifests are validated."""
from dataclasses import dataclass
from time import perf_counter
from .adaptation import AdaptationPolicy
from .contracts import GenerationOptions, Prediction, RunResult, StageError
from .storage import digest


@dataclass(frozen=True)
class RunConfig:
    options: GenerationOptions
    adaptation: AdaptationPolicy = AdaptationPolicy()
    allow_fixture: bool = False


class Runner:
    def __init__(self, store, adapter, editor, reconstructor=None):
        self.store, self.adapter, self.editor = store, adapter, editor
        self.reconstructor = reconstructor

    def run_source(self, request, config):
        from .contracts import Source3DRequest
        if type(request) is not Source3DRequest:
            raise TypeError('Expected Source3DRequest')
        return self._run(request, config, 'source3d')

    def run_video(self, request, config):
        from .contracts import VideoRequest
        if type(request) is not VideoRequest:
            raise TypeError('Expected VideoRequest')
        return self._run(request, config, 'video')

    def _run(self, request, config, condition):
        start, stage = perf_counter(), 'configuration'
        backends = {'editor': self.editor.identity, 'adapter': self.adapter.identity}
        if condition == 'video' and self.reconstructor:
            backends['reconstructor'] = self.reconstructor.identity
        fixture = any(b.fixture for b in backends.values())
        # Comparison contract deliberately excludes the differing input/ reconstruction backend.
        comparison = digest({'editor': self.editor.identity, 'input_spec': self.editor.input_spec,
                             'output_spec': self.editor.output_spec, 'options': config.options,
                             'adapter': self.adapter.identity, 'adaptation': config.adaptation})
        trace = {'schema_version': '1', 'condition_id': condition, 'key': request.key,
                 'request': request, 'config': config, 'backends': backends,
                 'comparison_id': comparison, 'fixture': fixture, 'artifacts': {}}
        trace['run_fingerprint'] = digest(trace)
        error, output_ref = None, None
        try:
            if fixture and not config.allow_fixture:
                raise ValueError('Fixture backend is disabled for real runs')
            if condition == 'video':
                if self.reconstructor is None:
                    raise ValueError('No reconstruction backend configured')
                stage = 'video_input'
                self.store.read_bytes(request.video.video)
                stage = 'reconstruction'
                reconstruction = self.reconstructor.reconstruct(request.video)
                trace['artifacts']['reconstruction'] = reconstruction
                self.store.read_bytes(reconstruction.diagnostics)
                if reconstruction.confidence:
                    self.store.read_bytes(reconstruction.confidence)
                motion = self.store.read_motion(reconstruction.motion)
            else:
                stage = 'source_input'
                motion = self.store.read_motion(request.source)
            stage = 'adaptation'
            adapted = self.adapter.adapt(motion, self.editor.input_spec, config.adaptation)
            adapted.motion.validate()
            if adapted.motion.spec != self.editor.input_spec:
                raise ValueError('Adapter output disagrees with editor input specification')
            self.store.read_bytes(adapted.transform_record)
            trace['artifacts']['adapted_motion'] = self.store.put_motion(adapted.motion)
            trace['artifacts']['transform_record'] = adapted.transform_record
            trace['quality_flags'] = adapted.quality_flags
            stage = 'editing'
            output = self.editor.edit(adapted.motion, request.instruction, config.options)
            stage = 'output_validation'
            output.validate()
            if output.spec != self.editor.output_spec:
                raise ValueError('Editor output specification mismatch')
            output_ref = self.store.put_motion(output)
            trace['artifacts']['prediction'] = output_ref
        except Exception as exc:
            error = StageError(stage, type(exc).__name__, str(exc))
        trace.update(status='failed' if error else 'succeeded', error=error,
                     elapsed_s=perf_counter() - start)
        trace_ref = self.store.put_json(trace)
        prediction = None if error else Prediction(request.key, condition, config.options.seed, output_ref, trace_ref)
        result = RunResult(request.key, condition, config.options.seed,
                           trace['status'], prediction, error, trace_ref)
        self.store.put_json(result)
        return result
