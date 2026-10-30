from reference_checkpoint import reference_checkpoint,clear_reference_checkpoints


def test_reference_identity_keys_and_explicit_reset(tmp_path):
    a=reference_checkpoint(tmp_path/'a')
    assert reference_checkpoint(str(tmp_path/'a')) is a
    assert reference_checkpoint(tmp_path/'b') is not a
    clear_reference_checkpoints()
    assert reference_checkpoint(tmp_path/'a') is not a
