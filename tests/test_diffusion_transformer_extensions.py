"""Behavioral checks for LDM, DiT and additive curation."""
import json
from pathlib import Path

import pytest

from p2i_kg.schema import KnowledgeGraph
from p2i_kg.store import GraphStore, seed_graph


def test_extension_is_additive_idempotent_and_preserves_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/ldm-dit.json').read_text())
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
    assert (result['added_nodes'], result['added_edges']) == (21, 55)
    assert store.load().nodes[0].summary == 'Preserve this user-curated text.'
    assert store.extend(fragment)['added_nodes'] == 0
    assert 'model:ldm' in {item['node']['id'] for item in
        store.load().query(concept='concept:latent_diffusion')}
    assert 'concept:adaptive_layer_norm' in store.load().effective_concepts('model:dit')


@pytest.mark.parametrize(('name', 'binding_count', 'target'), [
    ('ldm', 5, 'condition_encoder.2'),
    ('dit', 7, 'final_layer'),
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
    restored_output, harness_output = restored.build()(*inputs), harness.build()(*inputs)
    if isinstance(restored_output, tuple):
        for actual, expected in zip(restored_output, harness_output):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(restored_output, harness_output, rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid


def test_ldm_and_dit_outputs_and_gradient_paths():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model

    ldm, ldm_inputs = make_model('ldm')
    predicted, decoded = ldm(*ldm_inputs)
    assert predicted.shape == ldm_inputs[2].shape
    assert decoded.shape == ldm_inputs[0].shape
    ldm.denoising_loss(predicted, ldm_inputs[2]).backward()
    assert ldm.first_stage_encoder[0].weight.grad is not None
    assert ldm.condition_encoder[0].weight.grad is not None

    dit, dit_inputs = make_model('dit')
    output = dit(*dit_inputs)
    assert output.shape == dit_inputs[0].shape
    output.square().mean().backward()
    assert dit.patch_embed.weight.grad is not None
    assert dit.blocks[0].modulation[1].weight.grad is not None


def test_every_model_has_runtime_and_scope():
    models = [node for node in seed_graph().nodes if node.kind == 'model']
    assert len(models) == 49
    assert all(node.runtime_id and node.runtime_note for node in models)
