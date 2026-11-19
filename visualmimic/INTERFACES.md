# Module interfaces

| Boundary | Input | Output |
|---|---|---|
| Dataset to runner | Source3DRequest or VideoRequest with an instruction and sample key | Isolated source or video input |
| GVHMR reconstruction | VideoInput | Human motion, diagnostics and optional confidence |
| Motion adapter and editor | Validated motion, instruction and generation options | Edited human motion |
| GMR retargeter | Human motion and robot specification | Joint and root trajectories plus acceptance diagnostics |
| Reference compiler | Robot motion or compatible NPZ | Validated TrackingReference |
| SuperMimic executor | TrackingReference, frozen PolicyBundle and ExecutionProtocol | RobotRollout with states, actions and termination outcome |
| Evaluation | Predictions and held-out references, or a robot reference and rollout | Coverage and motion or tracking metrics |

The artifact store hashes payloads and restricts reads to configured roots. The runner records each backend and its options in the trace. A failed stage records its name and error. Test labels belong in the evaluation store.

[Original design and implementation notes](https://github.com/ShiminZhang/VisualMimic/blob/b18f29a3aeda040d1f96a45d79b90df8239d7e0d/visualmimic/INTERFACES.md)
