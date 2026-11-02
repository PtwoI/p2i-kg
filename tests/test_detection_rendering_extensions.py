"""Behavioral checks for set detection, radiance rendering and P2I recovery."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] / 'docs/detr-nerf.json').read_text())
    current = seed_graph().model_dump(mode='json')
    new_ids = {node['id'] for node in fragment['nodes']}
    new_edges = {(edge['source'], edge['relation'], edge['target']) for edge in fragment['edges']}
    current['nodes'] = [node for node in current['nodes'] if node['id'] not in new_ids]
    current['edges'] = [edge for edge in current['edges']
        if (edge['source'], edge['relation'], edge['target']) not in new_edges]
    current['nodes'][0]['summary'] = 'Preserve this user-curated text.'
    store = GraphStore(tmp_path / 'kg.db')
    store.replace(KnowledgeGraph.model_validate(current))
    result = store.extend(fragment)
    assert (result['added_nodes'], result['added_edges']) == (25, 57)
    assert store.load().nodes[0].summary == 'Preserve this user-curated text.'
    assert store.extend(fragment)['added_nodes'] == 0
    assert 'model:detr' in {item['node']['id'] for item in
        store.load().query(objective='objective:set_prediction_loss')}
    assert 'concept:cross_attention' in store.load().effective_concepts('model:detr')
    assert 'concept:volume_rendering' in store.load().effective_concepts('model:nerf')


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('detr', 6, 'class_head'),
    ('nerf', 5, 'density_head'),
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
    torch.testing.assert_close(restored.build()(*inputs), harness.build()(*inputs), rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid


def test_detr_parallel_slots_and_unique_bipartite_assignment():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    model, (images,) = make_model('detr')
    predictions = model(images)
    assert predictions.shape == (2, 5, 8)
    classes = torch.tensor([[0, 2], [1, 0]])
    boxes = torch.tensor([[[.1, .2, .3, .4], [.6, .5, .2, .1]],
                          [[.2, .3, .2, .2], [.7, .6, .1, .2]]])
    loss, assignment = model.set_loss(predictions, classes, boxes)
    assert all(len(set(pair)) == 2 for pair in assignment)
    loss.backward()
    assert model.query_embed.weight.grad is not None
    assert model.backbone.weight.grad is not None


def test_nerf_view_conditioning_and_occlusion_compositing():
    torch = pytest.importorskip('torch')
    from p2i_kg.models import VolumeRenderer
    from p2i_kg.runtime import make_model

    model, (points, directions, deltas) = make_model('nerf')
    rendered = model(points, directions, deltas)
    assert rendered.shape == (2, 3)
    assert not torch.equal(rendered, model(points, directions + 1., deltas))
    renderer = VolumeRenderer()
    rgb = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    density = torch.tensor([[[100.], [100.]]])
    out = renderer(rgb, density, torch.ones(1, 2))
    torch.testing.assert_close(out, torch.tensor([[1., 0., 0.]]), atol=1e-5, rtol=0)
    ((rendered - .5) ** 2).mean().backward()
    assert model.density_head.weight.grad is not None
    assert model.view_head[0].weight.grad is not None
