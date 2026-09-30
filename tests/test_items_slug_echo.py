import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError
from vmodel_contracts import ItemSpec
from spec_generator.graph import _heal_slugs, _items_plan_validator, _items_validator
from spec_generator.skeleton import ModelStructureError

REQS=[f"R-{n}" for n in range(1,16)]


def fixture():
 return json.loads((Path(__file__).parent/"fixtures/items-phi4-architecture-echo.json").read_text())


def test_exact_phi_plan_echo_fixture_passes_coverage_and_numbering():
 raw=fixture();before=copy.deepcopy(raw)
 normalized=_items_plan_validator(REQS)(raw)
 assert raw==before
 assert len(normalized)==5
 assert [item["slug"] for item in normalized]==["user-data-model","alert-data-model","disagreement-data-model","encryption-data-model","user-authentication-data-model"]
 assert [item["spec_id"] for item in normalized]==[f"SPEC-{n}" for n in range(1,6)]
 assert Counter(r for item in normalized for r in item["requirement_ids"])==Counter(REQS)
 assert _heal_slugs(_heal_slugs(raw))==_heal_slugs(raw)


def test_plan_only_fixture_does_not_invent_bulk_spec_bodies():
 with pytest.raises(ValidationError,match="body_markdown"):_items_validator(REQS)(fixture())


def test_shared_bulk_healer_preserves_authored_bodies_and_contract():
 raw=fixture()
 for item in raw:item["body_markdown"]=f"Test-authored body for {item['title']}"
 out=_items_validator(REQS)(raw)
 assert len(out)==5
 assert len({item["spec_id"] for item in out})==5
 assert Counter(r for item in out for r in item["requirement_ids"])==Counter(REQS)
 for item in out:ItemSpec.model_validate(item)


@pytest.mark.parametrize("slug",[None,"","architecture","constant"])
def test_missing_stage_or_repeated_slug_uses_title(slug):
 items=[{"slug":slug,"title":"User API"},{"slug":slug,"title":"Alert API"}]
 assert [item["slug"] for item in _heal_slugs(items)]==["user-api","alert-api"]


def test_title_slug_collision_gets_stable_suffix_without_merging_content():
 out=_heal_slugs([{"slug":"architecture","title":"User API","requirement_ids":["R-1"]},
                  {"slug":"architecture","title":"User API","requirement_ids":["R-2"]}])
 assert [item["slug"] for item in out]==["user-api","user-api-2"]
 assert len(out)==2


def test_exact_duplicate_comparison_preserves_array_order_and_any_content_difference():
 item={"slug":"architecture","title":"User","requirement_ids":["R-1","R-2"]}
 out=_heal_slugs([item,copy.deepcopy(item),{**item,"requirement_ids":["R-2","R-1"]}])
 assert len(out)==2


def test_dependency_remapping_after_exact_dedup_is_unambiguous():
 a={"slug":"User API","title":"User","requirement_ids":["R-1"]}
 out=_heal_slugs([a,copy.deepcopy(a),{"slug":"alerts","title":"Alerts","depends_on":["User API"]}])
 assert out[1]["depends_on"]==["user-api"]


def test_dependency_on_repeated_slug_is_rejected_not_guessed():
 with pytest.raises(ModelStructureError,match="ambiguous dependency"):
  _heal_slugs([{"slug":"architecture","title":"User"},{"slug":"architecture","title":"Alert"},
               {"slug":"events","title":"Events","depends_on":["architecture"]}])


@pytest.mark.parametrize("defect",["coverage","unknown_requirement","invalid_kind","missing_title","bad_slug","unknown_dependency"])
def test_broken_plans_still_rejected(defect):
 raw=fixture()[:5]
 if defect=="coverage":raw=raw[:1]
 elif defect=="unknown_requirement":raw[0]["requirement_ids"].append("R-99")
 elif defect=="invalid_kind":raw[0]["kind"]="invented"
 elif defect=="missing_title":raw[0]["title"]=""
 elif defect=="bad_slug":raw[0]["slug"]="!!!"
 else:raw[0]["depends_on"]=["not-a-spec"]
 with pytest.raises(ModelStructureError):_items_plan_validator(REQS)(raw)
