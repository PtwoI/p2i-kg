"""Behavioral checks for Perceiver, GraphSAGE and additive curation."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/perceiver-graphsage.json').read_text())
    current = seed_graph().model_dump(mode='json')
    new_ids = {node['id'] for node in fragment['nodes']}
    new_edges = {(edge['source'], edge['relation'], edge['target'])
        for edge in fragment['edges']}
    current['nodes'] = [node for node in current['nodes'] if node['id'] not in new_ids]
    current['edges'] = [edge for edge in current['edges']
        if (edge['source'], edge['relation'], edge['target']) not in new_edges]
    current['nodes'][0]['summary'] = 'Preserve this user-curated text.'
    store = GraphStore(tmp_path / 'kg.db')
    store.replace(KnowledgeGraph.model_validate(current))
    result = store.extend(fragment)
    assert (result['added_nodes'], result['added_edges']) == (22, 50)
    assert store.load().nodes[0].summary == 'Preserve this user-curated text.'
    assert store.extend(fragment)['added_nodes'] == 0
    assert 'model:graphsage' in {item['node']['id'] for item in
        store.load().query(objective='objective:graph_context_prediction')}
    assert 'concept:cross_attention' in store.load().effective_concepts('model:perceiver')
    assert 'concept:message_passing' in store.load().effective_concepts('model:graphsage')


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('perceiver', 5, 'classifier'),
    ('graphsage', 4, 'node_classifier'),
])
def test_bindings_edit_fresh_template_checkpoint_backward(name, binding_count, target):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.bindings import inspect_model
    from p2i_kg.runtime import make_model

    torch.set_num_threads(1)
    inspection = inspect_model(name)
    assert len(inspection['bindings']) == binding_count
    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.retrace(static=False).operations
    assert harness.apply(dict(type='set_parameter', target=target,
        parameter='bias', value=False)).committed
    fresh, _ = make_model(name)
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=fresh)
    assert restored.build().get_submodule(target).bias is None
    torch.testing.assert_close(restored.build()(*inputs), harness.build()(*inputs),
        rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid


def test_perceiver_keeps_fixed_latent_bottleneck_for_longer_input():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    model, (inputs,) = make_model('perceiver')
    seen = []
    hook = model.cross_attention.register_forward_pre_hook(
        lambda _module, args: seen.append((args[0].shape[1], args[1].shape[1])))
    try:
        assert model(inputs).shape == (2, 4)
        assert model(torch.randn(2, 37, 8)).shape == (2, 4)
    finally:
        hook.remove()
    assert seen == [(4, 12), (4, 37)]


def test_graphsage_mean_is_permutation_equivariant_and_backpropagates():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    model, (features, adjacency) = make_model('graphsage')
    output = model(features, adjacency)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = model(features[:, permutation],
        adjacency[:, permutation][:, :, permutation])
    torch.testing.assert_close(permuted, output[:, permutation])
    output.square().mean().backward()
    assert model.self_projection.weight.grad is not None
    assert model.neighbor_projection.weight.grad is not None
