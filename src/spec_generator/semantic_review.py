"""Apply reviewed semantic candidates to a checkpoint without model calls."""
from pathlib import Path
import json
from vmodel_contracts import SRSDocument
from vmodel_contracts.semantic_review import SemanticReview
from .graph import srs_document_data

def reviewed_srs(intent,original,proposal):
    base=SemanticReview.model_validate(original)
    proposed=SemanticReview.model_validate(proposal)
    if proposed.session_id!=intent['session_id'] or proposed.revision!=base.revision:
        raise ValueError('semantic review does not belong to the current candidate/session')
    if proposed.candidate!=base.candidate or proposed.findings!=base.findings:
        raise ValueError('semantic review evidence changed')
    return proposed.resolved_candidate(lambda data:SRSDocument.model_validate(srs_document_data(intent,data))),proposed

def load_reviewed_srs(intent,outdir,prior):
    original=(prior.get('semantic_reviews') or {}).get('srs')
    if not original or not original.get('findings'):return None
    path=Path(outdir)/'srs.semantic-review.json'
    if not path.is_file():raise ValueError('semantic candidate review is missing')
    return reviewed_srs(intent,original,json.loads(path.read_text()))
