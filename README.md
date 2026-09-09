# spec-generator

`IntentModel` in, V-model artifact bundle out: stakeholder needs, system
requirements, architecture, item-level specs, test hierarchy, AGENTS.md,
numbered build plan, manifest, traceability matrix, punch list.

Pipeline map: [PROCESS.md in the meta repo](https://github.com/ShyAlon/vmodel-spec-app/blob/main/docs/PROCESS.md) shows what each stage consumes and produces (and why stage 1 is "Formalizing", not "Extracting").

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
  needs.md  srs.md  architecture.md   # srs.md: the standalone, solution-agnostic SRS
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

Live per-stage progress lines (`-> Drafting architecture overview (stage 3/6)...`)
come from structured `ProgressEvent`s (progress.py) - the same stream the UI
will consume later (ADR-0010). Checkpoints persist to `bundle/checkpoints.sqlite`:
a crash mid-generation is recovered by rerunning the same command - completed
stages are never re-paid (ADR-0011).

## Boundary rule

Consumes only an `IntentModel` payload; emits only a `SpecBundle` (+ tests +
manifest). No imports from sibling modules.

## Local runs with Ollama (offline, $0) - THE DEFAULT

Every stage defaults to local Ollama: no API key, no cost, nothing leaves the
machine. Pull the model and run:

```bash
ollama pull qwen2.5-coder:14b
export OLLAMA_NUM_PARALLEL=1   # never load two models at once on 16GB
python -m spec_generator.cli out/<session>/intent_model.json -o bundle/
```

## Founder review gates (`--review`, ADR-0019)

Opt-in (`--review` on the CLI): the run PAUSES after the SRS and after the
architecture, renders the artifact (`gate_srs.md` / `gate_architecture.md` in
the output dir), and asks: **[a]ccept** continues; **[e]dit** opens the
stage's JSON in `$EDITOR` and re-validates it against the contract on save
(invalid edits reopen the editor - a hand edit can never corrupt downstream
state); **[f]eedback** takes one line and re-runs the stage with it applied;
**[r]eject** abandons the run (checkpointed - rerun the same command to
resume). A crash mid-gate just re-asks on rerun. Default runs (no flag) are
ungated, exactly as before.

Console output: each stage still gets one start/done line, and while a stage
runs, every item lands its own line as its JSON closes (`     - N-3 the system
shall ...`). The generation process itself is untouched - same one model call
per stage, same prompt, same completion; streaming only makes the items
visible as they arrive.

Model size: `qwen2.5-coder:14b` is the standing default - proven on a 16GB M4
with `OLLAMA_NUM_PARALLEL=1`. `qwen2.5-coder:7b` is the lighter fallback for
tighter machines:

```bash
for s in NEEDS SRS ARCHITECTURE ITEMS TESTS REVIEW; do
  export SPEC_GENERATOR_MODEL_$s=ollama:qwen2.5-coder:7b
done
```

## Remote providers (opt-in)

Route any stage to a strong remote model for founder-facing runs - generic
requirements are the free competitors' product:

```bash
export DEEPSEEK_API_KEY=...
for s in NEEDS SRS REVIEW; do
  export SPEC_GENERATOR_MODEL_$s=deepseek:deepseek-chat
done
```
