"""Additive ontology and executable checks for TCN, N-BEATS and TabNet."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/tcn-nbeats-tabnet.json').read_text())
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
    ('tcn', 4, 'sequence_head'),
    ('nbeats', 4, 'blocks.0.forecast_head'),
    ('tabnet', 4, 'decision_aggregation'),
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
    from p2i_kg.models import CausalConv1d
    from p2i_kg.runtime import make_model

    graph = seed_graph()
    models = [node for node in graph.nodes if node.kind == 'model']
    assert len(models) == 79
    assert all(node.runtime_id and node.runtime_note for node in models)

    causal = CausalConv1d(1, 1, kernel_size=3, dilation=2).eval()
    first = torch.randn(1, 1, 8)
    changed = first.clone()
    changed[..., 5:] += 100
    torch.testing.assert_close(causal(first)[..., :5], causal(changed)[..., :5])

    tcn, inputs = make_model('tcn')
    assert tcn(*inputs).shape == (2, 3)
    tcn(*inputs).sum().backward()
    assert tcn.temporal_blocks[0].causal_one.convolution.weight.grad is not None

    nbeats, inputs = make_model('nbeats')
    assert nbeats(*inputs).shape == (2, 4)
    nbeats(*inputs).square().mean().backward()
    assert nbeats.blocks[0].forecast_head.weight.grad is not None

    tabnet, inputs = make_model('tabnet')
    assert tabnet(*inputs).shape == (4, 4)
    tabnet(*inputs).square().mean().backward()
    assert tabnet.attentive_transformers[0].weight.grad is not None
