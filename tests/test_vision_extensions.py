"""Behavioral checks for architecture differences and independent edit recovery."""
import json
from pathlib import Path
import pytest
from p2i_kg.store import GraphStore, seed_graph
from p2i_kg.schema import KnowledgeGraph


def test_extension_preserves_existing_curation(tmp_path):
    fragment = json.loads((Path(__file__).parents[1]/'docs/mixer-mae.json').read_text())
    current = seed_graph().model_dump(mode='json')
    new_ids = {n['id'] for n in fragment['nodes']}
    new_edges = {(e['source'], e['relation'], e['target']) for e in fragment['edges']}
    current['nodes'] = [n for n in current['nodes'] if n['id'] not in new_ids]
    current['edges'] = [e for e in current['edges']
        if (e['source'],e['relation'],e['target']) not in new_edges
        and e['source'] not in new_ids and e['target'] not in new_ids
        and not (set(e['evidence']) & new_ids)]
    current['nodes'][0]['summary'] = 'User curated text must survive.'
    store = GraphStore(tmp_path/'kg.db')
    store.replace(KnowledgeGraph.model_validate(current))
    result = store.extend(fragment)
    assert (result['added_nodes'],result['added_edges']) == (15,34)
    assert store.load().nodes[0].summary == 'User curated text must survive.'
    assert store.extend(fragment)['added_nodes'] == 0
    assert 'concept:self_attention' not in store.load().effective_concepts('model:mixer')
    assert 'model:mae' in {x['node']['id'] for x in store.load().query(objective='objective:masked_pixel_mse')}


@pytest.mark.parametrize('name', ['mixer', 'mae'])
def test_bindings_edit_fresh_template_checkpoint_backward(name):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.runtime import make_model
    from p2i_kg.bindings import inspect_model
    torch.set_num_threads(1)
    inspection = inspect_model(name)
    assert len(inspection['bindings']) == (3 if name == 'mixer' else 4)
    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.retrace(static=False).operations
    target = 'channel_mixer.0' if name == 'mixer' else 'decoder_embed'
    assert harness.apply(dict(type='set_parameter', target=target, parameter='bias', value=False)).committed
    fresh, _ = make_model(name)
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=fresh)
    assert restored.build().get_submodule(target).bias is None
    torch.testing.assert_close(restored.build()(*inputs), harness.build()(*inputs), rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid


def test_mae_visible_only_and_masked_loss():
    torch = pytest.importorskip('torch')
    from p2i_kg.runtime import make_model
    torch.set_num_threads(1)
    model, (images,) = make_model('mae')
    seen = []
    handle = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    predictions = model(images)
    handle.remove()
    assert seen == [4] and predictions.shape == (2,16,48)
    # Alter only masked pixels: encoder input and resulting prediction must agree.
    changed = images.clone()
    for patch in model.masked.tolist():
        row, col = divmod(patch, 4)
        changed[:, :, row*4:row*4+4, col*4:col*4+4] += 10
    torch.testing.assert_close(model(changed), predictions, rtol=0, atol=0)
    noisy = predictions.clone()
    noisy[:,model.visible] += 10
    torch.testing.assert_close(model.reconstruction_loss(images,noisy), model.reconstruction_loss(images,predictions))
    model.reconstruction_loss(images,predictions).backward()
    assert model.mask_token.grad is not None and model.mask_token.grad.abs().sum() > 0
    assert model.patch_embed.weight.grad is not None
