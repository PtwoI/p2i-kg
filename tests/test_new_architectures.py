import pytest
import torch
from p2i_kg.runtime import make_model
from p2i_kg.bindings import inspect_model

@pytest.mark.parametrize('name,component', [('kan','component:kan_spline'), ('xlstm','component:xlstm_memory'), ('timemixer','component:timemixer_decomposition')])
def test_models_have_observed_bound_modules_and_gradients(name, component):
    model, inputs = make_model(name)
    model(*inputs).sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result = inspect_model(name)
    assert result['operations'] > 0
    assert component in {b['ontology_id'] for b in result['bindings']}
