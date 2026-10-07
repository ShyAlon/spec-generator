# Vendored contracts

`vmodel_contracts-0.27.0-py3-none-any.whl` was built from the clean
`ShyAlon/vmodel-contracts` revision
`865b8e7846b3c88d2a6a6c2baec94758c251591d` with `uv build --wheel`.

SHA-256: `ddbda9ddf6885c3b326e0777a22f785218ee919cc7f0b0cd599d01c429573bad`.

It is the same approved contracts source, not a fork. `uv.lock` records the
wheel hash. To update, build a clean approved contracts revision, record its
version/revision/hash here, update the source path and version in
`pyproject.toml`, then run `uv lock` and the deterministic suite. Commit the
wheel and lock together. Do not edit package contents inside the wheel.

Development uses `uv sync --locked --extra dev`. A pip-only install must
install this wheel before the project; pip does not read `[tool.uv.sources]`.
