"""Behavioral checks for SimCLR, BYOL and additive curation."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/simclr-byol.json').read_text())
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
    assert (result['added_nodes'], result['added_edges']) == (19, 47)
    assert store.load().nodes[0].summary == 'Preserve this user-curated text.'
    assert store.extend(fragment)['added_nodes'] == 0
    assert 'model:simclr' in {item['node']['id'] for item in
        store.load().query(objective='objective:nt_xent')}
    assert 'concept:momentum_target' in store.load().effective_concepts('model:byol')


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('simclr', 2, 'projector.2'),
    ('byol', 5, 'online_predictor.2'),
])
def test_bindings_edit_checkpoint_restore_backward(name, binding_count, target):
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


def test_simclr_nt_xent_and_byol_gradient_boundary():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    simclr, simclr_inputs = make_model('simclr')
    projections = simclr(*simclr_inputs)
    assert projections.shape == (4, 2, 8)
    simclr.nt_xent_loss(projections).backward()
    assert simclr.encoder.features[0].weight.grad is not None

    byol, byol_inputs = make_model('byol')
    outputs = byol(*byol_inputs)
    assert outputs.shape == (4, 4, 8)
    target_before = byol.target_encoder.features[0].weight.detach().clone()
    byol.bootstrap_loss(outputs).backward()
    assert byol.online_encoder.features[0].weight.grad is not None
    assert byol.target_encoder.features[0].weight.grad is None
    with torch.no_grad():
        byol.online_encoder.features[0].weight.add_(1.)
    byol.update_target(momentum=.5)
    assert not torch.equal(target_before, byol.target_encoder.features[0].weight)


def test_every_model_has_runtime_and_scope():
    graph = seed_graph()
    models = [node for node in graph.nodes if node.kind == 'model']
    assert len(models) == 52
    assert all(node.runtime_id and node.runtime_note for node in models)
