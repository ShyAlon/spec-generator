import json
from spec_generator.artifact_lifecycle import ArchitectureLifecycle
from vmodel_contracts import ArtifactState


def transitions(path):
 return [json.loads(x)["transition"] for x in path.read_text().splitlines()]


def test_architecture_partial_finalize_and_bounded_resume(tmp_path):
 baseline={"baseline_id":"b","mode":"greenfield","repositories":[]}
 life=ArchitectureLifecycle(tmp_path,"s",baseline)
 life.begin_or_checkpoint({"stack_keys":["language"]})
 life.begin_or_checkpoint({"stack_keys":["language"],"stack_entries":{}})
 event=life.finalize({"session_id":"s","stack_entries":[]}); life.commit_finalize(event)
 assert life.state==ArtifactState.COMPLETE
 life.resume("items")
 assert life.state==ArtifactState.PARTIAL
 assert transitions(life.path)==["begin","checkpoint","finalize","resume"]


def test_baseline_reanchor_is_stale_invalidate_restart(tmp_path):
 old={"baseline_id":"old","mode":"existing","repositories":[{"name":"x"}]}
 new={"baseline_id":"new","mode":"greenfield","repositories":[]}
 oldlife=ArchitectureLifecycle(tmp_path,"s",old)
 oldlife.bootstrap_complete({"architecture":"old"},old)
 life=ArchitectureLifecycle(tmp_path,"s",new)
 life.reanchor(["architecture","items","tests"])
 assert life.state==ArtifactState.PARTIAL
 assert transitions(life.path)[-3:]==["dependency_changed","invalidate","restart"]


def test_invalidated_or_stale_partial_cannot_be_read(tmp_path):
 baseline={"baseline_id":"b"};life=ArchitectureLifecycle(tmp_path,"s",baseline)
 life.bootstrap_complete({"a":1},baseline)
 # a changed dependency is recorded before invalidation; either state is barred
 event=life._event(__import__('vmodel_contracts').LifecycleTransition.DEPENDENCY_CHANGED,ArtifactState.STALE,
  "changed",{"changed_dependency":"anchor"})
 __import__('vmodel_contracts').append_lifecycle_event(life.path,event)
 try: life.guard_partial_read(); assert False
 except ValueError as e: assert "stale" in str(e)


def test_complete_architecture_reopens_for_review_regeneration(tmp_path):
 baseline={"baseline_id":"b","mode":"greenfield","repositories":[]}
 life=ArchitectureLifecycle(tmp_path,"s",baseline)
 life.begin_or_checkpoint({"plan":1})
 event=life.finalize({"architecture":"complete"});life.commit_finalize(event)
 life.regenerate("expert requested regeneration")
 assert life.state==ArtifactState.PARTIAL
 life.begin_or_checkpoint({"new_plan":1})
 assert transitions(life.path)[-3:]==["invalidate","restart","checkpoint"]


def test_crashed_node_reentry_reopens_complete_on_first_partial(tmp_path):
 baseline={"baseline_id":"b","mode":"greenfield","repositories":[]}
 life=ArchitectureLifecycle(tmp_path,"s",baseline)
 life.begin_or_checkpoint({"old_plan":1})
 event=life.finalize({"architecture":"complete"});life.commit_finalize(event)
 # Exact continuation shape: no gate callback survives the crash. The node
 # starts directly and its first durable action is a partial checkpoint.
 life.checkpoint_reentry({"new_plan":1})
 assert life.state==ArtifactState.PARTIAL
 assert transitions(life.path)[-3:]==["invalidate","restart","checkpoint"]


def sealed_continuation(tmp_path, *, legacy=False):
 from spec_generator.artifact_lifecycle import stable_hash
 old={"contract_version":"0.19.0","baseline_id":"b"}
 new={**old,"contract_version":"0.20.0"}
 content={"architecture":"sealed"}
 life=ArchitectureLifecycle(tmp_path,"s",old)
 life.bootstrap_complete(content,old)
 if legacy: life.anchor_hash=stable_hash(new)
 life.resume("items")
 return ArchitectureLifecycle(tmp_path,"s",old),content,new


def test_same_resume_retry_is_noop_and_backfill_is_append_only(tmp_path):
 life,content,new=sealed_continuation(tmp_path)
 before=life.path.read_bytes()
 life.resume("items",content=content)
 assert life.path.read_bytes()==before
 life.commit_finalize(life.finalize(content))
 assert life.state==ArtifactState.COMPLETE
 assert life.path.read_bytes().startswith(before)
 assert transitions(life.path)==["begin","finalize","resume","finalize"]


def test_prefixed_stamp_drift_retry_uses_proven_compatible_anchor(tmp_path):
 from spec_generator.artifact_lifecycle import stable_hash
 life,content,new=sealed_continuation(tmp_path,legacy=True)
 before=life.path.read_bytes()
 life.resume("items",content=content,compatible_anchor_hashes=(stable_hash(new),))
 assert life.path.read_bytes()==before
 life.commit_finalize(life.finalize(content))
 assert life.recorded_anchor_hash==life.anchor_hash


import pytest

@pytest.mark.parametrize("change",["destination","content","anchor","no_content"])
def test_resume_retry_rejects_different_continuation(tmp_path,change):
 life,content,new=sealed_continuation(tmp_path,legacy=change=="anchor")
 before=life.path.read_bytes()
 with pytest.raises(ValueError):
  life.resume("gate_architecture" if change=="destination" else "items",
              content=None if change=="no_content" else {"architecture":"changed"} if change=="content" else content)
 assert life.path.read_bytes()==before


def test_true_partial_cannot_be_resumed_as_completed(tmp_path):
 life=ArchitectureLifecycle(tmp_path,"s",{"baseline_id":"b"})
 life.begin_or_checkpoint({"plan":1})
 before=life.path.read_bytes()
 with pytest.raises(ValueError):life.resume("items",content={"architecture":"sealed"})
 assert life.path.read_bytes()==before


@pytest.mark.parametrize("state",["stale","invalidated","migrated"])
def test_resume_rejects_other_noncomplete_states(tmp_path,state):
 from vmodel_contracts import LifecycleTransition,append_lifecycle_event
 life=ArchitectureLifecycle(tmp_path,"s",{"baseline_id":"b"})
 life.bootstrap_complete({"architecture":"sealed"},{"baseline_id":"b"})
 if state=="stale":
  event=life._event(LifecycleTransition.DEPENDENCY_CHANGED,ArtifactState.STALE,"changed",{"changed_dependency":"anchor"})
 elif state=="invalidated":
  event=life._event(LifecycleTransition.INVALIDATE,ArtifactState.INVALIDATED,"changed",{"named_boundary":"architecture","downstream_closure_computed":True})
 else:
  event=life._event(LifecycleTransition.MIGRATE,ArtifactState.MIGRATED,"migrated",{"approved_migrator":True,"source_hash":"old","source_version":"0.19.0","target_schema_valid":True})
 append_lifecycle_event(life.path,event)
 before=life.path.read_bytes()
 with pytest.raises(ValueError):life.resume("items",content={"architecture":"sealed"})
 assert life.path.read_bytes()==before
