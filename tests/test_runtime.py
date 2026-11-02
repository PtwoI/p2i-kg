import json
from urllib.request import urlopen
import pytest

p2i = pytest.importorskip('p2i')
torch = pytest.importorskip('torch')
from fastapi.testclient import TestClient
from p2i_kg.app import create_app
from p2i_kg.runtime import make_model, RUNTIME_IDS
from p2i.checkpoint import checkpoint_bytes


@pytest.mark.parametrize('name', RUNTIME_IDS)
def test_model_factories_edit_checkpoint_backward(name):
    if name in ('resnet18','vit'):pytest.importorskip('torchvision')
    if name in ('bert','gpt2'):pytest.importorskip('transformers')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, inputs = make_model(name)
        harness = p2i.Harness(model, example_inputs=inputs)
        assert harness.retrace(static=False).operations
        restored = p2i.Harness.load(checkpoint_bytes(harness), model=model)
        torch.testing.assert_close(restored.build()(*inputs), model(*inputs), rtol=0, atol=0)
        report = restored.validate(backward=True)
        assert report.backward_valid is (name != 'perceptron')
        if name == 'perceptron':
            assert set(model(*inputs).flatten().tolist()) <= {0.,1.}
        # Catalog factories have different head names. Exercise a real editable
        # linear projection rather than assuming every model exposes `head`.
        editable = {node.qualified_name for node in restored.architecture().nodes if node.editable}
        target = next(path for path, module in reversed(list(model.named_modules()))
                      if path in editable and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)))
        action = {'type':'set_parameter','target':target,'parameter':'bias','value':model.get_submodule(target).bias is None}
        assert restored.apply(action).committed
        assert restored.validate(backward=True).backward_valid is (name != 'perceptron')
        rebuilt = p2i.Harness.load(checkpoint_bytes(restored), model=restored.build())
        torch.testing.assert_close(rebuilt.build()(*inputs), restored.build()(*inputs), rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)


def test_real_p2i_session_and_download(tmp_path):
    with TestClient(create_app(tmp_path/'kg.sqlite3')) as client:
        response = client.post('/api/kg/runtime/model:mlp', json={})
        assert response.status_code == 200, response.text
        url = response.json()['url']
        with urlopen(url+'/api/model/manifest', timeout=15) as opened:
            manifest = json.load(opened)
        assert manifest['counts']['operations'] > 0
        assert client.post('/api/kg/runtime/model:mlp', json={}).json()['reused']
        downloaded = client.get('/api/kg/models/model:lstm/checkpoint')
        assert downloaded.status_code == 200
        assert p2i.load_model(downloaded.content).recurrent.hidden_size == 16
        assert client.delete('/api/kg/runtime/model:mlp').json()['closed']
        assert not client.delete('/api/kg/runtime/model:mlp').json()['closed']
        assert client.post('/api/kg/runtime/model:missing',json={}).status_code == 404


def test_ontology_module_bindings_are_real():
    from p2i_kg.bindings import inspect_model
    result = inspect_model('seq2seq')
    bindings = {b['ontology_id']: b for b in result['bindings']}
    assert bindings['component:cross_attention']['module_path'] == 'decoder.0.multihead_attn'
    assert bindings['component:cross_attention']['managed_by']
    assert bindings['component:decoder']['parameters']['dim_feedforward'] == 64
    assert result['operations'] > 0


@pytest.mark.parametrize('name,expected', [('deepsets','component:set_sum_pool'),
    ('pointnet','component:symmetric_max_pool')])
def test_set_model_bindings_and_permutation_invariance(name, expected):
    from p2i_kg.bindings import inspect_model
    model, inputs = make_model(name)
    permutation = (torch.tensor([4, 0, 6, 1, 3, 2, 5])
        if name == 'deepsets' else torch.randperm(16))
    torch.testing.assert_close(model(*inputs), model(inputs[0][:, permutation]))
    result = inspect_model(name)
    assert expected in {binding['ontology_id'] for binding in result['bindings']}
    assert result['operations'] > 0
