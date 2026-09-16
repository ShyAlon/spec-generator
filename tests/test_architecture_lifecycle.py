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
