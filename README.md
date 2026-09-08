# spec-generator

`IntentModel` in, V-model artifact bundle out: stakeholder needs, system
requirements, architecture, item-level specs, test hierarchy, AGENTS.md,
numbered build plan, manifest, traceability matrix, punch list.

## Design line (hybrid B+C)

- **LLM stages**: needs, requirements, architecture, item specs, test
  derivation, two independent review passes. Authoring quality is where a
  CTO-level tester notices rigor - these keep a strong model.
- **Deterministic tail** (`assembly.py`, `render.py`): id numbering, the
  traceability matrix, milestone ordering (topological over `depends_on`),
  the manifest, AGENTS.md, and hard validation. An orphan requirement or an
  untested acceptance criterion fails assembly and lands on the punch list.
  Structure is code; judgment is LLM. Never the reverse.

Runs as a **background job** (Vercel's 300s cap - see ARCHITECTURE.md
ADR-0006); the LangGraph checkpointer makes a multi-minute pipeline
resumable. Batch-API eligible stages are marked in `model_config.py`.

## Bundle layout (Spec Kit-compatible)

```
bundle/
  AGENTS.md            # build order, decided architecture, guardrails
  manifest.json        # machine-readable index - read this first
  needs.md  requirements.md  architecture.md
  specs/SPEC-*.md      # item-level specs
  tests.json           # AcceptanceTest set for verification-loop
  traceability.json    # requirement -> specs -> tests
  usage.jsonl          # per-stage token/cost records
```

## Run

```bash
pip install -e .[dev]
export DEEPSEEK_API_KEY=...
python -m spec_generator.cli out/<session>/intent_model.json -o bundle/
```

## Boundary rule

Consumes only an `IntentModel` payload; emits only a `SpecBundle` (+ tests +
manifest). No imports from sibling modules.
