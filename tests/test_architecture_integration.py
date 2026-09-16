

def test_founder_tech_extraction_rejects_ordinary_word_collisions():
    from spec_generator.graph import _founder_named_techs
    intent={"tech_preferences":[{"area":"language","preference":"Agnostic","strength":"preferred","source_turns":[12]}],
            "stakeholder_needs":[
                {"statement":"A claims clerk accepts or declines the claim"},
                {"statement":"It goes to the insurer"},
                {"statement":"The agent should react in one go"},
            ]}
    assert _founder_named_techs(intent) == []


def test_founder_tech_extraction_keeps_explicit_need_choice():
    from spec_generator.graph import _founder_named_techs
    intent={"tech_preferences":[],"stakeholder_needs":[
        {"statement":"The app should use free resources like Supabase where possible"}]}
    assert _founder_named_techs(intent) == ["supabase"]
