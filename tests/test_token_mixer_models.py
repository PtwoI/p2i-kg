import pytest


@pytest.mark.parametrize('name,edit_path,bindings', [
    ('poolformer', 'head', {'block.mixer', 'block.channel_mlp'}),
    ('convmixer', 'head', {'block.spatial', 'block.channel'}),
    ('resmlp', 'head', {'block.affine1', 'block.token_mix', 'block.channel_mlp'}),
])
def test_token_mixers_trace_edit_restore_and_backward(name, edit_path, bindings):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.bindings import inspect_model
    from p2i_kg.runtime import make_model

    inspected = inspect_model(name)
    assert inspected['operations'] > 0
    assert {binding['module_path'] for binding in inspected['bindings']} == bindings
    observed = {module['qualified_name'] for module in inspected['observed']['modules']
                if module['executed']}
    assert bindings <= observed

    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.apply(dict(type='set_parameter', target=edit_path, parameter='bias',
                              value=False)).committed
    assert harness.retrace(static=False).operations
    fresh, _ = make_model(name)
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=fresh)
    assert restored.build().get_submodule(edit_path).bias is None
    torch.testing.assert_close(restored.build()(*inputs), harness.build()(*inputs), rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
