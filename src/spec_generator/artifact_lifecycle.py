"""Architecture artifact lifecycle pilot.

The host PostgreSQL checkpoint retains workflow position. This adapter gates architecture partial
and artifact reads/writes with the canonical vmodel-contracts transition table
and appends the resulting evidence to artifact-lifecycle.jsonl.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vmodel_contracts import (ArtifactState, LifecycleEvent, LifecycleTransition,
                              append_lifecycle_event as _append_lifecycle_event)


def append_lifecycle_event(path,event):
    from .stop_fence import guarded_publish
    from .document_runtime import configured,append
    if configured():
        from vmodel_contracts import validate_transition
        typed=validate_transition(event)
        guarded_publish(lambda:append(path,(typed.model_dump_json()+"\n").encode()))
        return typed
    return guarded_publish(lambda:_append_lifecycle_event(path,event))


def stable_hash(value) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


class ArchitectureLifecycle:
    def __init__(self, outdir: Path, session_id: str, anchor: dict):
        self.path = outdir / "artifact-lifecycle.jsonl"
        self.artifact_id = f"architecture:{session_id}"
        self.anchor_hash = stable_hash(anchor)

    def _events(self) -> list[dict]:
        from .document_runtime import configured,_bound,read_text
        if configured():
            documents,key=_bound(self.path)
            return documents.recent_lifecycle_events(key,self.artifact_id)
        if not self.path.is_file():return []
        events=[]
        for line in read_text(self.path).splitlines():
            if line.strip():
                value=json.loads(line)
                if value.get("artifact_id")==self.artifact_id:
                    events.append(value)
        return events

    @property
    def state(self) -> ArtifactState:
        events=self._events()
        return ArtifactState(events[-1]["to_state"]) if events else ArtifactState.MISSING

    @property
    def recorded_anchor_hash(self) -> str | None:
        events=self._events()
        return events[-1].get("anchor_hash") if events else None

    def _event(self, transition, to_state, reason, guards, *, content=None, invalidates=()):
        event=LifecycleEvent(
            artifact_id=self.artifact_id, artifact_kind="architecture",
            transition=transition, from_state=self.state, to_state=to_state,
            actor="spec-generator", reason=reason,
            content_hash=stable_hash(content) if content is not None else None,
            anchor_hash=self.anchor_hash, producer_stage="architecture",
            invalidates=list(invalidates), guard_evidence=guards)
        return event

    def bootstrap_complete(self, content, anchor: dict) -> None:
        """Adopt a pre-pilot checkpoint into shadow history without changing it."""
        if self.state != ArtifactState.MISSING:
            return
        current_anchor=self.anchor_hash
        self.anchor_hash=stable_hash(anchor)
        append_lifecycle_event(self.path,self._event(LifecycleTransition.BEGIN,ArtifactState.PARTIAL,
            "pre-pilot architecture checkpoint adopted",{"producer_owns_artifact":True,"inputs_recorded":True,"anchor_recorded":True}))
        event=self._event(LifecycleTransition.FINALIZE,ArtifactState.COMPLETE,
            "pre-pilot architecture was schema-valid and complete",{"schema_valid":True,"complete":True,"hashes_sealed":True},content=content)
        append_lifecycle_event(self.path,event)
        self.anchor_hash=current_anchor

    def begin_or_checkpoint(self, content) -> None:
        if self.state == ArtifactState.MISSING:
            e=self._event(LifecycleTransition.BEGIN,ArtifactState.PARTIAL,
                "architecture partial started",{"producer_owns_artifact":True,"inputs_recorded":True,"anchor_recorded":True})
        elif self.state == ArtifactState.PARTIAL:
            e=self._event(LifecycleTransition.CHECKPOINT,ArtifactState.PARTIAL,
                "architecture partial advanced",{"monotonic_progress":True,"same_inputs":True,"same_anchor":True})
        else:
            raise ValueError(f"architecture partial write forbidden while lifecycle state is {self.state.value}")
        append_lifecycle_event(self.path,e)

    def checkpoint_reentry(self, content) -> None:
        """Checkpoint a running architecture node, reopening a sealed prior run.

        A graph crash can persist the node's restart intent while the lifecycle
        has already sealed the preceding architecture. Re-entering the producer
        and attempting its first partial is the durable restart signal.
        """
        if self.state == ArtifactState.COMPLETE:
            self.regenerate("architecture node re-entered after a sealed run")
        self.begin_or_checkpoint(content)

    def guard_partial_read(self) -> None:
        if self.state in (ArtifactState.INVALIDATED, ArtifactState.STALE, ArtifactState.MIGRATED):
            raise ValueError(f"architecture partial read forbidden while lifecycle state is {self.state.value}")

    def finalize(self, content) -> LifecycleEvent | None:
        if self.state == ArtifactState.COMPLETE:
            if self.recorded_anchor_hash != self.anchor_hash:
                raise ValueError("complete architecture cannot be written under a changed anchor")
            return None  # idempotent backfill/render, no fake transition
        if self.state == ArtifactState.MISSING:
            append_lifecycle_event(self.path,self._event(LifecycleTransition.BEGIN,ArtifactState.PARTIAL,
                "architecture generation began and completed without intermediate partial",{"producer_owns_artifact":True,"inputs_recorded":True,"anchor_recorded":True}))
        return self._event(LifecycleTransition.FINALIZE,ArtifactState.COMPLETE,
            "architecture contract validated and sealed",{"schema_valid":True,"complete":True,"hashes_sealed":True},content=content)

    def commit_finalize(self, event: LifecycleEvent | None) -> None:
        if event is not None:
            append_lifecycle_event(self.path,event)

    def resume(self, next_artifact: str, *, content=None, compatible_anchor_hashes=()) -> None:
        """Replay only an evidenced continuation of the same sealed artifact.

        The caller validates checkpoint content and proves any alternate anchor
        hashes represent stamp-only drift. No shared transition is weakened.
        """
        events = self._events()
        if self.state == ArtifactState.PARTIAL:
            last = events[-1]
            guards = last.get("guard_evidence") or {}
            if (last.get("transition") != LifecycleTransition.RESUME.value
                    or guards.get("named_next_artifact") != next_artifact
                    or guards.get("legal_outgoing_arc") != f"architecture->{next_artifact}"):
                raise ValueError("partial architecture is not the same recorded continuation")
            seals = [event for event in events[:-1]
                     if event.get("transition") == LifecycleTransition.FINALIZE.value]
            seal = seals[-1] if seals else None
            if (seal is None or events[-2] != seal or content is None
                    or seal.get("content_hash") != stable_hash(content)
                    or seal.get("anchor_hash") != self.anchor_hash):
                raise ValueError("resume retry does not match the sealed checkpoint architecture")
            allowed = {self.anchor_hash, *compatible_anchor_hashes}
            if last.get("anchor_hash") not in allowed:
                raise ValueError("resume retry has a changed anchor")
            return  # Already recorded; validated backfill appends FINALIZE.
        append_lifecycle_event(self.path,self._event(LifecycleTransition.RESUME,ArtifactState.PARTIAL,
            f"bounded architecture stop resumed toward {next_artifact}",
            {"named_next_artifact":next_artifact,"legal_outgoing_arc":f"architecture->{next_artifact}"}))

    def regenerate(self, reason: str) -> None:
        """Reopen a sealed architecture for an explicit review-gate rewrite."""
        if self.state == ArtifactState.COMPLETE:
            append_lifecycle_event(self.path, self._event(
                LifecycleTransition.INVALIDATE, ArtifactState.INVALIDATED,
                reason, {"named_boundary": "architecture", "downstream_closure_computed": True},
                invalidates=["architecture", "items", "tests", "assembly", "manifest", "traceability"]))
        if self.state == ArtifactState.INVALIDATED:
            append_lifecycle_event(self.path, self._event(
                LifecycleTransition.RESTART, ArtifactState.PARTIAL,
                "architecture producer restarted after review feedback",
                {"producer_rerun_started": True, "current_inputs": True}))
        elif self.state not in (ArtifactState.MISSING, ArtifactState.PARTIAL):
            raise ValueError(f"architecture regeneration cannot start from lifecycle state {self.state.value}")

    def reanchor(self, invalidates: list[str]) -> None:
        old_anchor=self.recorded_anchor_hash
        if self.state == ArtifactState.COMPLETE:
            append_lifecycle_event(self.path,self._event(LifecycleTransition.DEPENDENCY_CHANGED,ArtifactState.STALE,
                "SystemBaseline anchor hash changed",{"changed_dependency":f"anchor:{old_anchor}->{self.anchor_hash}"},invalidates=invalidates))
        if self.state == ArtifactState.STALE:
            append_lifecycle_event(self.path,self._event(LifecycleTransition.INVALIDATE,ArtifactState.INVALIDATED,
                "architecture and downstream closure invalidated for new baseline",
                {"named_boundary":"architecture","downstream_closure_computed":True},invalidates=invalidates))
        if self.state == ArtifactState.INVALIDATED:
            append_lifecycle_event(self.path,self._event(LifecycleTransition.RESTART,ArtifactState.PARTIAL,
                "architecture producer restarted under current baseline",
                {"producer_rerun_started":True,"current_inputs":True},invalidates=invalidates))
