"""Additive ontology and executable checks for Neural ODE, FNO and RWKV."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


NEW = ('neuralode', 'fno', 'rwkv')


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/neuralode-fno-rwkv.json').read_text())
    current = seed_graph().model_dump(mode='json')
    new_ids = {node['id'] for node in fragment['nodes']}
    new_edges = {(edge['source'], edge['relation'], edge['target'])
        for edge in fragment['edges']}
    current['nodes'] = [node for node in current['nodes'] if node['id'] not in new_ids]
    current['edges'] = [edge for edge in current['edges']
        if (edge['source'], edge['relation'], edge['target']) not in new_edges
        and edge['source'] not in new_ids and edge['target'] not in new_ids
        and not (set(edge['evidence']) & new_ids)]
    current['nodes'][0]['summary'] = 'Preserve this user-curated text.'
    store = GraphStore(tmp_path / 'kg.db')
    store.replace(KnowledgeGraph.model_validate(current))
    result = store.extend(fragment)
    assert (result['added_nodes'], result['added_edges']) == (
        len(fragment['nodes']), len(fragment['edges']))
    assert store.load().nodes[0].summary == 'Preserve this user-curated text.'
    assert store.extend(fragment)['added_nodes'] == 0


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('neuralode', 4, 'classifier'),
    ('fno', 4, 'projection.2'),
    ('rwkv', 5, 'lm_head'),
])
def test_trace_edit_checkpoint_restore_backward(name, binding_count, target):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.bindings import inspect_model
    from p2i_kg.runtime import make_model

    torch.set_num_threads(1)
    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.retrace(static=False).operations
    assert harness.apply({'type':'set_parameter', 'target':target,
        'parameter':'bias', 'value':False}).committed
    fresh, _ = make_model(name)
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=fresh)
    torch.testing.assert_close(restored.build()(*inputs), harness.build()(*inputs),
                               rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    inspection = inspect_model(name)
    assert inspection['operations'] > 0
    assert len(inspection['bindings']) == binding_count


def test_architecture_specific_behavior_and_coverage():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model
    graph = seed_graph()
    models = [node for node in graph.nodes if node.kind == 'model']
    assert len(models) == 55
    assert all(node.runtime_id and node.runtime_note for node in models)

    neuralode, inputs = make_model('neuralode')
    neuralode(*inputs).sum().backward()
    assert neuralode.vector_field.network[0].weight.grad is not None
    fno, inputs = make_model('fno')
    assert fno(*inputs).shape == (2, 1, 32)
    fno(*inputs).square().mean().backward()
    assert fno.spectral_convolution.weight.grad is not None
    rwkv, inputs = make_model('rwkv')
    output = rwkv(*inputs)
    assert output.shape == (2, 10, 32)
    output.square().mean().backward()
    assert rwkv.time_mix.time_decay.grad is not None
