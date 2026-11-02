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
        target = {'perceptron':'linear','mlp':'layers.0','cnn':'features.0','resnet18':'fc',
            'vit':'heads.head','bert':'pooler.dense','detr':'class_head',
            'nerf':'density_head','perceiver':'classifier',
            'graphsage':'node_classifier'}.get(name,'head')
        action = {'type':'set_parameter','target':target,'parameter':'bias','value':False}
        if name == 'gpt2':action = {'type':'set_parameter','target':'transformer.drop','parameter':'p','value':0.1}
        assert restored.apply(action).committed
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
