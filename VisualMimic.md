# Experiment design

For each MotionFix source and edit instruction, compare a 3D-input condition (A) with a video-input condition (B). Render the source to video for B and give its reconstruction to the same editor used for A. Fix the generation options, GMR settings, SuperMimic checkpoint and simulator protocol across the pair. Keep target motions for evaluation only.

Inspect the edited human motion, the retargeted G1 reference (C for the A branch), and the simulated G1 executions. Record the number of paired samples, missing outputs, early terminations and uncertain judgments alongside success rates. This makes losses from reconstruction, editing, retargeting and execution easier to investigate.

Use the original MotionFix split and keep all renders of a source together. The historical import audit found 1,177 exact segments shared across official splits and a missing training annotation for sample 006722.

[Original design and implementation notes](https://github.com/ShiminZhang/VisualMimic/blob/b18f29a3aeda040d1f96a45d79b90df8239d7e0d/VisualMimic.md)
