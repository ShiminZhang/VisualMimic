# Policy and world model

The policy tracks a reference using proprioception and a learned motion embedding. The world model predicts transitions for a 71-dimensional state. The imagination trainer rolls the model forward over a short horizon and differentiates the actor objective through the predicted states. See `src/supermimic/world_model/`, `policy/` and `imagination/` for the implementation.

[Full original research record](https://github.com/ShiminZhang/VisualMimic/blob/b18f29a3aeda040d1f96a45d79b90df8239d7e0d/supermimic/docs/PAPER_SPEC.md)
