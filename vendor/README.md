# Vendored contracts

`vmodel_contracts-0.27.2-py3-none-any.whl` was built from the clean
`ShyAlon/vmodel-contracts` revision
`a649a4ea4bd80f930e5fe07ed94fe74886a42837` with `uv build --wheel`.

SHA-256: `7081549ad8cb67f392d755f01fc9342bfa2223b369ba0dc3c393df9cd3e2a98d`.

It is the same approved contracts source, not a fork. `uv.lock` records the
wheel hash. To update, build a clean approved contracts revision, record its
version/revision/hash here, update the source path and version in
`pyproject.toml`, then run `uv lock` and the deterministic suite. Commit the
wheel and lock together. Do not edit package contents inside the wheel.

Development uses `uv sync --locked --extra dev`. A pip-only install must
install this wheel before the project; pip does not read `[tool.uv.sources]`.
