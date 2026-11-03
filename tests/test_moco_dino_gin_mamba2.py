"""Behavioral and P2I checks for the four incremental architecture models."""
import json
from pathlib import Path

import pytest

p2i = pytest.importorskip('p2i')
torch = pytest.importorskip('torch')

from p2i.checkpoint import checkpoint_bytes
from p2i_kg.bindings import BINDINGS, inspect_model
from p2i_kg.store import seed_graph
from p2i_kg.runtime import make_model


NEW = ('moco', 'dino', 'gin', 'mamba2')


def test_extension_is_additive_and_all_models_remain_executable():
    graph = seed_graph()
    models = [node for node in graph.nodes if node.kind == 'model']
    assert len(models) == 46
    assert {f'model:{name}' for name in NEW} <= {node.id for node in models}
    assert all(node.runtime_id and node.runtime_note for node in models)
    fragment = json.loads((Path(__file__).parents[1] /
        'docs/moco-dino-gin-mamba2.json').read_text())
    assert len(fragment['nodes']) == 37
    assert len(fragment['edges']) == 52


@pytest.mark.parametrize('name', NEW)
def test_new_runtime_trace_edit_restore_backward_and_bindings(name):
    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.retrace(static=False).operations
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=model)
    torch.testing.assert_close(restored.build()(*inputs), model(*inputs), rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    editable = {node.qualified_name for node in restored.architecture().nodes if node.editable}
    target = next(path for path, module in reversed(list(model.named_modules()))
        if path in editable and isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)))
    module = model.get_submodule(target)
    assert restored.apply({'type':'set_parameter','target':target,'parameter':'bias',
        'value':module.bias is None}).committed
    rebuilt = p2i.Harness.load(checkpoint_bytes(restored), model=restored.build())
    torch.testing.assert_close(rebuilt.build()(*inputs), restored.build()(*inputs),
        rtol=0, atol=0)
    inspection = inspect_model(name)
    assert inspection['operations'] > 0
    assert len(inspection['bindings']) == len(BINDINGS[name])


def test_moco_dino_gradient_boundaries_and_gin_permutation():
    moco, inputs = make_model('moco')
    torch.nn.functional.cross_entropy(moco(*inputs), torch.zeros(4, dtype=torch.long)).backward()
    assert moco.query_encoder.features[0].weight.grad is not None
    assert moco.key_encoder.features[0].weight.grad is None
    dino, inputs = make_model('dino')
    outputs = dino(*inputs)
    dino.self_distillation_loss(outputs).backward()
    assert dino.student_encoder.features[0].weight.grad is not None
    assert dino.teacher_encoder.features[0].weight.grad is None
    gin, inputs = make_model('gin')
    permutation = torch.tensor([3, 0, 4, 1, 2])
    features, adjacency = inputs
    permuted_adjacency = adjacency[:, permutation][:, :, permutation]
    torch.testing.assert_close(gin(features, adjacency),
        gin(features[:, permutation], permuted_adjacency))
