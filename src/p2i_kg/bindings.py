"""Verified factory-specific links from ontology components to P2I modules."""
from .runtime import make_model

# Paths refer to the fixed, versioned educational factories. They are verified
# against the actual architecture, not guessed for arbitrary uploaded models.
BINDINGS = {
    'seq2seq': [('component:encoder', 'encoder.0'), ('component:decoder', 'decoder.0'),
        ('component:positions', 'positions'), ('component:cross_attention', 'decoder.0.multihead_attn'),
        ('component:self_attention', 'encoder.0.self_attn'), ('component:ffn', 'decoder.0.linear1')],
    'transformer': [('component:self_attention', 'blocks.0.attention'), ('component:ffn', 'blocks.0.mlp')],
    'bert': [('component:encoder', 'encoder.layer.0'), ('component:self_attention', 'encoder.layer.0.attention.self'),
        ('component:ffn', 'encoder.layer.0.intermediate'), ('component:positions', 'embeddings.position_embeddings')],
    'vit': [('component:patches', 'conv_proj'), ('component:encoder', 'encoder.layers.encoder_layer_0'),
        ('component:self_attention', 'encoder.layers.encoder_layer_0.self_attention'), ('component:ffn', 'encoder.layers.encoder_layer_0.mlp')],
    'gpt2': [('component:self_attention', 'transformer.h.0.attn'), ('component:ffn', 'transformer.h.0.mlp'),
        ('component:positions', 'transformer.wpe')],
    'cnn': [('component:convolution', 'features.0')],
    'resnet18': [('component:convolution', 'layer1.0.conv1')],
    **{key: [('component:recurrent', 'recurrent')] for key in ('rnn', 'lstm', 'gru')},
}


def inspect_model(runtime_id):
    import p2i
    model, inputs = make_model(runtime_id)
    harness = p2i.Harness(model, example_inputs=inputs)
    observed = harness.retrace(static=False)
    architecture = harness.architecture()
    paths = {node.qualified_name: node for node in architecture.nodes}
    bindings = []
    for ontology_id, path in BINDINGS.get(runtime_id, []):
        node = paths[path]  # A changed factory must fail, not emit a broken link.
        module = model.get_submodule(path)
        bindings.append({'ontology_id': ontology_id, 'module_path': path,
            'architecture_node_id': node.id, 'module_type': node.module_type,
            'parameters': node.parameters, 'editable': node.editable,
            'managed_by': node.managed_by,
            'parameter_count': sum(p.numel() for p in module.parameters())})
    return {'runtime_id': runtime_id, 'p2i_version': p2i.__version__,
        'operations': len(observed.operations), 'modules': len(architecture.nodes),
        'bindings': bindings, 'architecture': architecture.model_dump(mode='json'),
        'scope': 'Current fixed example configuration, with randomly initialized weights.'}
