

def test_adr_title_renderer_owns_single_id_prefix():
    from spec_generator.render import _adr_title
    assert _adr_title("ADR-3", "ADR-3: Use PostgreSQL") == "Use PostgreSQL"
    assert _adr_title("ADR-3", "Use PostgreSQL") == "Use PostgreSQL"
