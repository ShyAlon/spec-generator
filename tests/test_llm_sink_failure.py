import logging
from spec_generator import model_config as mc


def test_invalid_llm_sink_warns_once_without_path_content_or_credentials(tmp_path,monkeypatch,caplog):
 monkeypatch.setattr(mc,'_LLM_SINK_FAILURES',set(),raising=False)
 monkeypatch.setattr(mc,'_llm_log_sink',lambda:str(tmp_path))
 with caplog.at_level(logging.WARNING):
  for _ in range(3):mc._log_llm({'event':'fixture','password':'private-value','raw':'private project'})
 messages=[r.message for r in caplog.records if 'llm.log.degraded' in r.message]
 assert len(messages)==1
 assert 'IsADirectoryError' in messages[0]
 assert str(tmp_path) not in messages[0] and 'private' not in messages[0]


def test_disabled_sink_is_not_a_failure(monkeypatch,caplog):
 monkeypatch.setattr(mc,'_llm_log_sink',lambda:None)
 with caplog.at_level(logging.WARNING):mc._log_llm({'event':'fixture'})
 assert not any('llm.log.degraded' in r.message for r in caplog.records)
