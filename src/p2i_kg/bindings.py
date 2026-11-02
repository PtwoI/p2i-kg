"""Verified factory-specific links from ontology components to P2I modules."""
from .runtime import make_model

# Paths refer to the fixed, versioned educational factories. They are verified
# against the actual architecture, not guessed for arbitrary uploaded models.
BINDINGS = {
    'perceiver': [('component:perceiver_input_projection', 'input_projection'),
        ('component:latent_array', 'latent_array'),
        ('component:perceiver_cross_attention', 'cross_attention'),
        ('component:latent_processor', 'latent_processor'),
        ('component:perceiver_classifier', 'classifier')],
    'graphsage': [('component:neighbor_aggregator', 'aggregator'),
        ('component:self_projection', 'self_projection'),
        ('component:neighbor_projection', 'neighbor_projection'),
        ('component:node_classifier', 'node_classifier')],
    'detr': [('component:detr_backbone', 'backbone'), ('component:detr_encoder', 'encoder'),
        ('component:detr_decoder', 'decoder'), ('component:object_queries', 'query_embed'),
        ('component:detection_heads', 'class_head'), ('component:detection_heads', 'box_head')],
    'nerf': [('component:coordinate_encoder', 'position_encoder'), ('component:radiance_mlp', 'point_mlp'),
        ('component:density_head', 'density_head'), ('component:view_color_head', 'view_head'),
        ('component:volume_renderer', 'volume_renderer')],
    'mixer': [('component:patches', 'patch_embed'), ('component:token_mixer', 'token_mixer'),
        ('component:channel_mixer', 'channel_mixer')],
    'mae': [('component:patches', 'patch_embed'), ('component:mae_encoder', 'encoder'),
        ('component:mae_decoder', 'decoder'), ('component:pixel_predictor', 'head')],
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
    'vgg': [('component:convolution', 'features')],
    'inception': [('component:convolution', 'inception3a')],
    'densenet': [('component:convolution', 'features.denseblock1')],
    'mobilenet': [('component:convolution', 'features.1')],
    'efficientnet': [('component:convolution', 'features.1')],
    'convnext': [('component:convolution', 'features.1')],
    'unet': [('component:convolution', 'encoder')],
    'vae': [('component:latent_encoder', 'encoder'), ('component:latent_decoder', 'decoder')],
    'gan': [('component:generator', 'generator'), ('component:discriminator', 'discriminator')],
    'ddpm': [('component:denoiser', 'denoiser')],
    'gcn': [('component:graph_aggregation', 'aggregation')],
    'gat': [('component:graph_aggregation', 'attention')],
    't5': [('component:encoder', 'backbone.encoder'), ('component:decoder', 'backbone.decoder')],
    'swin': [('component:patches', 'features.0'), ('component:self_attention', 'features.1.0.attn')],
    'roformer': [('component:self_attention', 'encoder.layer.0.attention.self')],
    'llama': [('component:self_attention', 'model.layers.0.self_attn'), ('component:ffn', 'model.layers.0.mlp')],
    'clip': [('component:image_encoder', 'backbone.vision_model'), ('component:text_encoder', 'backbone.text_model')],
    'mamba': [('component:state_space', 'layers.0.mixer')],
    'switch': [('component:router', 'backbone.encoder.block.0.layer.1.mlp.router'),
               ('component:expert', 'backbone.encoder.block.0.layer.1.mlp.experts.expert_0')],
    **{key: [('component:recurrent', 'recurrent')] for key in ('rnn', 'lstm', 'gru')},
}


def inspect_model(runtime_id):
    import p2i
    model, inputs = make_model(runtime_id)
    harness = p2i.Harness(model, example_inputs=inputs)
    observed = harness.retrace(static=False, capture_values=True)
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
        'observed': observed.model_dump(mode='json'),
        'inspection': harness.inspection.as_dict(),
        'configuration': harness.observed_configuration,
        'scope': 'Current fixed example configuration, with randomly initialized weights.'}
