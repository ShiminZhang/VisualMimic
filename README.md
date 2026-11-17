# VisualMimic code snapshot

Source-only snapshot of VisualMimic and the local SuperMimic reproduction.

- `visualmimic/`: reconstruction, motion editing, retargeting, execution, evaluation, and comparison viewers.
- `tests/`: VisualMimic tests.
- `supermimic/`: policy, world model, training, simulation, and evaluation implementation. See its README and LICENSE.
- `dependencies.json`: upstream repositories and exact revisions; apply any listed dependency patches after checkout into `visualmimic/external/<name>`.

This snapshot excludes credentials, datasets, licensed body models, checkpoints, videos, environments, and run artifacts. Those dependencies must be provisioned separately. Existing scripts may contain ICE-specific paths and require local configuration. Snapshot packaging does not establish that every module or experiment passes.

See `visualmimic/README.md` and `visualmimic/INTERFACES.md` for implementation details. No new license is assigned to code whose licensing was unspecified; upstream licenses continue to apply.
