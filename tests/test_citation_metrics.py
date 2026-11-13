import importlib.util,json
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('citation_metrics',Path(__file__).parents[1]/'scripts/citation-metrics.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

def test_missing_records_are_unknown_not_zero(tmp_path):
 result=MODULE.summarize(tmp_path)
 assert result['quote_copy_rejections'] is None and result['usage_rows_by_stage'] is None

def test_existing_records_are_counted_without_invented_retry_costs(tmp_path):
 (tmp_path/'spec').mkdir();(tmp_path/'rejected-llm-outputs').mkdir()
 rows=[{'stage':'architecture','reason':'cite the recorded decision text','attempt':1,'duration_s':27.3},{'stage':'architecture','reason':'cite the recorded decision text','attempt':2,'duration_s':27.81}]
 (tmp_path/'rejected-llm-outputs/manifest.jsonl').write_text(json.dumps(rows))
 (tmp_path/'spec/usage.jsonl').write_text('{"stage":"architecture"}\n{"stage":"srs"}\n')
 (tmp_path/'spec/run.log').write_text('Drafting architecture overview FAILED after 1236.7s\n')
 result=MODULE.summarize(tmp_path)
 assert result['quote_copy_rejections']==2 and result['quote_copy_repair_rejections']==1
 assert result['quote_copy_rejected_call_seconds']==[27.3,27.81]
 assert result['observed_stage_log_durations'][0]['elapsed_s']==1236.7
 assert result['usage_rows_by_stage']=={'srs':1,'architecture':1}
