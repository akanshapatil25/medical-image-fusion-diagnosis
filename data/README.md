# Data directory

- `raw/` stores the downloaded BRISC archive and extracted official dataset. Keep these local; do not commit the data.
- `splits/brisc/` stores verified train/validation/test CSV manifests and the integrity report.
- `demo/` stores synthetic MRI-like images used only to demonstrate software behavior. Training/evaluation explicitly rejects demo paths.
- `processed/` is reserved for derived images when needed.

BRISC's official test partition must remain separate from training and validation. MRI–CT fusion requires genuine paired studies and does not use this MRI-only dataset.
