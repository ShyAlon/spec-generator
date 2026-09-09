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

Written INCREMENTALLY as each stage completes (atomic tmp+rename): needs.*
after stage 1, srs.* after stage 2 (re-written if review fixes apply),
architecture.* after stage 3, items.json + specs/ after stage 4, tests.json
after stage 5. A late failure leaves every completed stage's artifacts on
disk; AGENTS.md / manifest.json / traceability.json land at assembly. On
resume, artifacts for stages completed before this shipped are backfilled
from checkpoint state at startup (state is authoritative, zero stages re-run).

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

Structured output by default (local runs): every stage's call goes to Ollama's
native schema-constrained generation (`format` = a JSON schema DERIVED from
vmodel-contracts, so the contracts stay the single source of truth). The model
literally cannot emit invalid JSON shape. Grammar enforces shape only - not
content quality and not length - so tolerant extraction + one retry stays as
the net, and the quality comparison is yours to eyeball once:
`python scripts/probe_structured.py out/<session>/intent_model.json`
(runs the needs stage with and without the schema on your Ollama, prints
validity, wall time, and both outputs). Needs Ollama >= 0.5. Opt out:
`export SPEC_GENERATOR_STRUCTURED=0`.

Reliability: every stage tolerantly extracts JSON (fences, prose-wrapped) and
auto-retries ONCE with the failure as feedback when the model returns invalid
JSON - a transient flake recovers without you. Only a second consecutive
failure stops the run, with a raw-output snippet for diagnosis; rerun the same
command and the checkpoint resumes at that stage. The tests stage (largest
payload) gets extra output-token headroom against truncation.

Console output: each stage still gets one start/done line, and while a stage
runs, every item lands its own line as its JSON closes (`     - N-3 the system
shall ...`). The generation process itself is untouched - same one model call
per stage, same prompt, same completion; streaming only makes the items
visible as they arrive.

Determinism (ADR-0022): the model owns prose; code owns everything
enumerable. All stages default to temperature 0 with a fixed seed
(`SPEC_GENERATOR_SEED`, default 42) so a failure replays exactly. The SRS and
items stages run as PER-SLOT MICRO-CALLS: purpose/scope in one small call,
one call per requirement fill, then a spec plan (references enum-constrained
to the real requirement ids) plus one body call per spec. A bad fill fails
one cheap call - never a long composite payload - and each slot/spec lands
its own console line as it completes. The old one-bulk-call-per-stage shape
stays available if wall time is worse on your hardware:

```bash
export SPEC_GENERATOR_BULK=1   # one composite call per stage (pre-ADR-0022)
```

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

