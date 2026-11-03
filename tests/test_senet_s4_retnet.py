"""Behavioral and P2I recovery checks for SENet, S4 and RetNet."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


NEW = ('senet', 's4', 'retnet')


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/senet-s4-retnet.json').read_text())
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
    assert {'model:senet', 'model:s4', 'model:retnet'} <= {
        node.id for node in store.load().nodes}


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('senet', 5, 'head'),
    ('s4', 3, 'head'),
    ('retnet', 5, 'output_projection'),
])
def test_trace_edit_checkpoint_restore_backward(name, binding_count, target):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.bindings import inspect_model
    from p2i_kg.runtime import make_model

    torch.set_num_threads(1)
    inspection = inspect_model(name)
    assert inspection['operations'] > 0
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


def test_architecture_specific_computation_and_gradients():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    senet, (images,) = make_model('senet')
    assert senet(images).shape == (2, 4)
    senet(images).sum().backward()
    assert senet.block.se.excitation[0].weight.grad is not None

    s4, (sequence,) = make_model('s4')
    assert s4(sequence).shape == (2, 4)
    s4(sequence).sum().backward()
    assert s4.state_kernel.log_decay.grad is not None

    retnet, (tokens,) = make_model('retnet')
    output = retnet(tokens)
    assert output.shape == tokens.shape
    changed_future = tokens.clone()
    changed_future[:, 8:] += 100.
    torch.testing.assert_close(retnet(tokens)[:, :8], retnet(changed_future)[:, :8])
    output.square().mean().backward()
    assert retnet.retention.qkv.weight.grad is not None


def test_every_model_has_unique_runtime_and_scope():
    graph = seed_graph()
    models = [node for node in graph.nodes if node.kind == 'model']
    assert len(models) == 52
    runtime_ids = [node.runtime_id for node in models]
    assert all(runtime_ids) and len(runtime_ids) == len(set(runtime_ids))
    assert all(node.runtime_note for node in models)
