"""JEV-D10 rule 3: Jev grounding adapter, failure matrix with a mocked provider."""
import pytest

from spec_generator import grounding as g
from spec_generator.findings_ledger import Ledger, revision_of

BODY = ("# SPEC-1\n\nThe evaluator scores each attempt from expected and observed note events and returns "
        "pitch and rhythm categories immediately.\n\nIf scoring fails the attempt is kept and retried.")
REQS = [{"requirement_id": "R-13", "statement": "The system shall score attempts.", "acceptance": [{"criterion_id": "AC-1", "statement": "Score is shown."}]}]
EXCL = [{"id": "E1", "text": "No accounts."}]


class Mock:
    def __init__(self, q1="yes", q2="no", q3="no", body="B1", basis="R-13", fail=None, mutate=None):
        self.q1, self.q2, self.q3, self.body, self.basis, self.fail, self.mutate, self.calls = q1, q2, q3, body, basis, fail, mutate, 0

    def classify(self, state, questions, model):
        self.calls += 1
        if self.fail:
            raise self.fail
        ans = {}
        for k in questions:
            q = k.split(".")[1]
            if k.endswith(".body"):
                first = next(c for c in questions[k][1] if c.startswith("B"))
                ans[k] = (first if self.body == "B1" else self.body) if getattr(self, q) == "yes" else "NONE"
            elif k.endswith(".basis"):
                ans[k] = self.basis if getattr(self, q) == "yes" or q == "q1" else "UNKNOWN"
            else:
                ans[k] = getattr(self, q)
        if self.mutate:
            ans = self.mutate(ans)
        return g.ProviderResponse(ans, "typesafe", "jev-x", 10, 2, 0.1)


def run(provider, **kw):
    return g.ground_spec("SPEC-1", kw.pop("body", BODY), REQS, provider, exclusions=kw.pop("exclusions", EXCL), **kw)


def test_favorable_is_eligible_and_quotes_are_code_materialized():
    env = run(Mock())
    a = env["per_requirement"][0]
    assert env["status"] == g.ELIGIBLE and env["eligible"] and env["check_complete"]
    assert a["body_quote"] in BODY and a["body_quote"].startswith("The evaluator") and a["evidence"][0]["sha256"]


def test_q1_no_raises_a_semantic_finding_and_blocks():
    env = run(Mock(q1="no"))
    assert env["status"] == g.ELIGIBLE and not env["eligible"]
    ledger = Ledger(); ids = g.record_grounding(ledger, env)
    assert len(ids) == 1 and ledger.open_findings()[0]["rule_id"] == "grounding.jev" and ledger.gate()["blocked"]


def test_q2_yes_needs_body_and_basis_and_raises():
    env = run(Mock(q2="yes", body="B1", basis="R-13"))
    assert env["per_requirement"][0]["q2"] == "yes"
    ledger = Ledger(); assert g.record_grounding(ledger, env)
    assert run(Mock(q2="yes", basis="UNKNOWN"))["status"] == g.UNVERIFIED


def test_q3_yes_raises_and_unknown_is_unverified():
    assert g.record_grounding(Ledger(), run(Mock(q3="yes", basis="E1")))
    env = run(Mock(q3="unknown"))
    assert env["status"] == g.UNVERIFIED and not env["eligible"]


def test_q1_has_no_unknown():
    assert run(Mock(q1="unknown"))["status"] == g.UNVERIFIED   # not an allowed Q1 choice


@pytest.mark.parametrize("err", [g.ProviderUnavailable("no key"), TimeoutError("t"), ConnectionError("net")])
def test_unavailable_never_falls_back(err):
    env = run(Mock(fail=err))
    assert env["status"] == g.UNAVAILABLE and not env["eligible"]
    ledger = Ledger(); g.record_grounding(ledger, env)
    assert ledger.gate()["blocked"] and ledger.gate()["incomplete"][0]["status"] == g.UNAVAILABLE


@pytest.mark.parametrize("mutate", [
    lambda a: {**a, "R-13.q1": "maybe"},                      # invalid enum
    lambda a: {k: v for k, v in a.items() if k != "R-13.q2"},  # missing key
    lambda a: {**a, "R-13.extra": "yes"},                      # extra key
    lambda a: {**a, "R-13.q1.body": "B99"},                    # invented evidence id
])
def test_malformed_payload_is_unverified_and_preserved(mutate):
    env = run(Mock(mutate=mutate))
    assert env["status"] == g.UNVERIFIED and env["returned_payload"] and not env["eligible"]


def test_stale_body_hash():
    env = run(Mock(), expect_body_sha256="deadbeef")
    assert env["status"] == g.STALE and not env["eligible"]


def test_disabled_is_not_checked_never_a_pass():
    env = run(None)
    assert env["status"] == g.NOT_CHECKED and not env["eligible"]
    ledger = Ledger(); g.record_grounding(ledger, env)
    assert ledger.history[-1]["status"] == g.NOT_CHECKED


def test_empty_or_thin_body_is_q1_no_without_calling_jev():
    m = Mock()
    env = run(m, body="# SPEC-1\n")
    assert m.calls == 0 and env["per_requirement"][0]["q1"] == "no" and not env["eligible"]


def test_oversized_body_is_incomplete_not_a_pass():
    big = "\n\n".join(f"Paragraph {i} describes behavior." for i in range(g.MAX_BODY_SPANS_PER_CALL * g.MAX_CHUNKS_PER_REQUIREMENT + 1))
    env = run(Mock(), body=big)
    assert env["status"] == g.NOT_CHECKED and "exceeds" in env["reason"]
    ledger = Ledger(); g.record_grounding(ledger, env); assert ledger.gate()["blocked"]


def test_chunked_body_is_covered_and_yes_in_any_chunk_counts():
    paras = "\n\n".join(f"Paragraph {i} describes behavior." for i in range(g.MAX_BODY_SPANS_PER_CALL + 5))
    m = Mock(); env = run(m, body=paras)
    assert m.calls == 2 and env["status"] == g.ELIGIBLE


def test_cache_hits_on_formatting_only_edit_and_misses_on_content_edit():
    cache = g.GroundingCache(); m = Mock()
    run(m, cache=cache); run(m, body=BODY.replace("# SPEC-1", "## SPEC-1").replace(". ", ".  "), cache=cache)
    assert m.calls == 1
    run(m, body=BODY + "\n\nIt also exports a file.", cache=cache)
    assert m.calls == 2


def test_gray_echo_question_is_advisory_and_unavailable_leaves_finding_open():
    finding = {"detail": {"check": {"matched_span": "Wait up to 30 seconds"}}}
    assert g.ask_gray_echo(Mock(fail=g.ProviderUnavailable("x")), finding, BODY)["status"] == g.UNAVAILABLE
    assert g.ask_gray_echo(None, finding, BODY)["status"] == g.NOT_CHECKED

    class Echo:
        def classify(self, s, q, m): return g.ProviderResponse({"echo": "yes"}, "typesafe", "m")
    assert g.ask_gray_echo(Echo(), finding, BODY)["answer"] == "yes"


def test_typesafe_provider_requires_sdk_and_key():
    with pytest.raises(g.ProviderUnavailable):
        g.TypeSafeGroundingProvider(api_key=None)


def test_pipeline_runs_grounding_and_blocks_assembly_on_q1_no(tmp_path):
    """Review-boundary wiring: Jev says NO -> semantic finding -> assembly blocked; disabled -> NOT CHECKED, no block."""
    from langgraph.checkpoint.memory import MemorySaver
    from conftest import SYSTEM_BASELINE
    from spec_generator.errors import AssemblyBlocked
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_progress import INTENT
    import test_review_fixes_and_persistence as t
    from spec_generator import model_config
    import pytest as _p
    mp = _p.MonkeyPatch()
    try:
        box = {}
        t.QueueModel.QUEUES = {"review": [[]]}
        from conftest import ModelFactoryLedger, SchemaRoutingArchitectureModel
        led = ModelFactoryLedger(SchemaRoutingArchitectureModel)
        mp.setattr(model_config, "build_chat_model",
                   lambda stage, **kw: led.build(stage, kw.get("schema")) if stage == "architecture" else box.setdefault(stage, t.QueueModel(stage)))
        graph = build_graph(UsageLog("s"), checkpointer=MemorySaver(), grounding=Mock(q1="no"))
        cfg = {"configurable": {"thread_id": "gr"}}
        with pytest.raises(AssemblyBlocked):
            graph.invoke({"intent_model": dict(INTENT), "system_baseline": SYSTEM_BASELINE}, cfg)
        ledger = Ledger.from_dict(graph.get_state(cfg).values["findings_ledger"])
        assert any(f["rule_id"] == "grounding.jev" for f in ledger.open_findings())
    finally:
        mp.undo()


def test_missing_exclusions_make_q3_unknown_not_a_pass():
    env = run(Mock(), exclusions=[])
    assert env["per_requirement"][0]["q3"] == "unknown"
    assert env["status"] == g.UNVERIFIED and not env["eligible"]
    ledger = Ledger(); g.record_grounding(ledger, env)
    assert ledger.gate()["blocked"]
