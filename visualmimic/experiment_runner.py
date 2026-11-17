"""Complete stage-one robot trial orchestration."""
from dataclasses import dataclass
from .contracts import StageError
from .storage import digest
from .tracking_evaluation import evaluate_tracking
from .instruction_evaluation import evaluate_instruction


@dataclass(frozen=True)
class ExperimentSpec:
    generation: object
    adaptation: object
    retarget: object
    execution: object
    output_fps: float = 50.
    instruction_task: object = None


class VisualMimicRunner:
    def __init__(self, store, normalizer, editor, retargeter, compiler, executor,
                 robot, reconstructor=None, renderer=None):
        self.store, self.normalizer, self.editor = store, normalizer, editor
        self.retargeter, self.compiler, self.executor = retargeter, compiler, executor
        self.robot, self.reconstructor, self.renderer = robot, reconstructor, renderer

    def run_source(self, request, experiment):
        from .contracts import Source3DRequest
        if type(request) is not Source3DRequest: raise TypeError('Expected source request')
        return self._run(request, experiment, 'source3d')

    def run_video(self, request, experiment):
        from .contracts import VideoRequest
        if type(request) is not VideoRequest: raise TypeError('Expected video request')
        return self._run(request, experiment, 'video')

    def _run(self, request, experiment, branch):
        result = {'key': request.key, 'branch': branch, 'request': request,
                  'experiment': experiment, 'policy': self.executor.bundle,
                  'editor': self.editor.identity, 'robot': self.robot,
                  'status': 'running', 'stages': {}, 'reports': {}}
        result['trial_id'] = digest(result)
        stage = 'source'
        try:
            if branch == 'video':
                stage = 'reconstruction'
                if self.reconstructor is None: raise ValueError('Video backend not configured')
                self.store.read_bytes(request.video.video)
                reconstructed = self.reconstructor.reconstruct(request.video)
                result['stages'][stage] = reconstructed
                motion = self.store.read_motion(reconstructed.motion)
            else:
                motion = self.store.read_motion(request.source)
            stage = 'normalization'
            normalized = self.normalizer.adapt(motion, self.editor.input_spec, experiment.adaptation)
            source = normalized.motion.validate()
            result['stages'][stage] = {'motion': self.store.put_motion(source), 'transforms': normalized.transform_record}
            stage = 'editing'
            edited = self.editor.edit(source, request.instruction, experiment.generation).validate()
            result['stages'][stage] = self.store.put_motion(edited)
            executions = {}
            for name, motion in [('baseline', source), ('edited', edited)]:
                stage = name + '_retarget'
                retargeted = self.retargeter.retarget(motion, self.robot, experiment.retarget)
                result['stages'][stage] = {'mapping_id': retargeted.mapping_id,
                                           'diagnostics': retargeted.diagnostics, 'status': retargeted.status,
                                           'reasons': retargeted.reasons}
                if retargeted.status != 'accepted': raise ValueError(f'Retarget rejected: {retargeted.reasons}')
                stage = name + '_compile'
                reference = self.compiler.compile(retargeted.motion, experiment.output_fps)
                result['stages'][stage] = reference
                stage = name + '_execute'
                rollout = self.executor.execute(reference, experiment.execution)
                result['stages'][stage] = rollout
                executions[name] = rollout
                stage = name + '_tracking_evaluation'
                result['reports'][name + '_tracking'] = evaluate_tracking(reference, rollout, self.store)
                if self.renderer:
                    stage = name + '_render'
                    result['stages'][stage] = self.renderer(rollout, name, result['trial_id'])
            stage = 'instruction_evaluation'
            result['reports']['instruction'] = evaluate_instruction(request.instruction, executions['baseline'],
                                                                     executions['edited'], experiment.instruction_task, self.store)
            # Orchestration finishing does not mean instruction success or rollout completion.
            result['status'] = 'executed'
            result['robot_outcomes'] = {k: r.outcome for k, r in executions.items()}
        except Exception as exc:
            result['status'] = 'failed'
            result['error'] = StageError(stage, type(exc).__name__, str(exc))
        return self.store.put_json(result)
