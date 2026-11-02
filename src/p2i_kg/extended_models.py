"""Small executable architecture examples. No pretrained downloads or weights."""
import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(nn.Conv2d(incoming, incoming, 3, stride=stride, padding=1, groups=incoming, bias=False),
                         nn.BatchNorm2d(incoming), nn.ReLU(),
                         nn.Conv2d(incoming, outgoing, 1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU())


class TinyMobileNetV1(nn.Module):
    """Original MobileNet depthwise/pointwise schedule, width multiplier .25."""
    def __init__(self):
        super().__init__()
        layers = [nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU())]
        incoming = 8
        schedule = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
                    *[(512, 1)] * 5, (1024, 2), (1024, 1)]
        for channels, stride in schedule:
            outgoing = channels // 4
            layers.append(DepthwiseSeparable(incoming, outgoing, stride))
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, 4)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.ReLU())


class TinyUNet(nn.Module):
    """Two-level U-Net with skip concatenation; spatial padding replaces cropping."""
    def __init__(self, incoming=3, outgoing=2):
        super().__init__()
        self.encoder = DoubleConv(incoming, 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(8, 16)
        self.upsample = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.decoder = DoubleConv(16, 8)
        self.head = nn.Conv2d(8, outgoing, 1)

    def forward(self, x):
        skip = self.encoder(x)
        hidden = self.bottleneck(self.pool(skip))
        return self.head(self.decoder(torch.cat([skip, self.upsample(hidden)], dim=1)))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.mean = nn.Linear(12, 4)
        self.log_variance = nn.Linear(12, 4)
        self.decoder = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 16), nn.Sigmoid())

    def forward(self, x, epsilon):
        encoded = self.encoder(x)
        mean, log_variance = self.mean(encoded), self.log_variance(encoded)
        latent = mean + torch.exp(0.5 * log_variance) * epsilon
        return self.decoder(latent), mean, log_variance


class TinyGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 16), nn.Tanh())
        self.discriminator = nn.Sequential(nn.Linear(16, 12), nn.LeakyReLU(0.2), nn.Linear(12, 1), nn.Sigmoid())

    def forward(self, noise, real):
        generated = self.generator(noise)
        return generated, self.discriminator(generated), self.discriminator(real)


class TinyDDPM(nn.Module):
    """One q(x_t|x_0) noise step and time-conditioned epsilon prediction."""
    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8), nn.SiLU())
        self.image_projection = nn.Conv2d(1, 8, 3, padding=1)
        self.denoiser = TinyUNet(8, 1)
        self.register_buffer('alpha_bar', torch.cumprod(1 - torch.linspace(.0001, .02, 10), 0))

    def forward(self, clean, noise, timestep):
        alpha = self.alpha_bar[timestep].reshape(-1, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        condition = self.time_embedding(timestep).unsqueeze(-1).unsqueeze(-1)
        return self.denoiser(self.image_projection(noisy) + condition)


class GraphConvolution(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.projection = nn.Linear(incoming, outgoing, bias=False)

    def forward(self, x, adjacency):
        linked = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        inverse = linked.sum(-1).clamp_min(1).rsqrt()
        normalized = inverse[:, None] * linked * inverse[None, :]
        return normalized @ self.projection(x)


class TinyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregation = GraphConvolution(8, 12)
        self.head = GraphConvolution(12, 3)

    def forward(self, x, adjacency):
        return self.head(F.relu(self.aggregation(x, adjacency)), adjacency)


class GraphAttention(nn.Module):
    """Single-head additive neighbor attention with self-loops."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 12, bias=False)
        self.source_score = nn.Linear(12, 1, bias=False)
        self.target_score = nn.Linear(12, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adjacency):
        hidden = self.projection(x)
        scores = self.activation(self.source_score(hidden) + self.target_score(hidden).transpose(0, 1))
        mask = (adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)) > 0
        weights = self.softmax(scores.masked_fill(~mask, -1e9))
        return weights @ hidden


class TinyGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GraphAttention()
        self.head = nn.Linear(12, 3)

    def forward(self, x, adjacency):
        return self.head(F.elu(self.attention(x, adjacency)))


class Seq2SeqLogits(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, source, target):
        return self.backbone(input_ids=source, decoder_input_ids=target, use_cache=False, return_dict=False)[0]


class CLIPLogits(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, tokens, images):
        return self.backbone(input_ids=tokens, pixel_values=images, return_dict=False)[0]



def make_extended(name):
    if name == 'mobilenet':
        return TinyMobileNetV1().eval(), (torch.randn(1, 3, 32, 32),)
    if name in ('vgg', 'inception', 'densenet', 'efficientnet', 'convnext', 'swin'):
        from torchvision import models
        if name == 'vgg':
            model = models.vgg11(weights=None, num_classes=4)
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.classifier = nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 4))
            size = 32
        elif name == 'inception':
            model = models.googlenet(weights=None, num_classes=4, aux_logits=False, init_weights=False)
            size = 64
        elif name == 'densenet':
            model = models.DenseNet(growth_rate=8, block_config=(1, 1, 1, 1), num_init_features=16, num_classes=4)
            size = 32
        elif name == 'efficientnet':
            model = models.efficientnet_b0(weights=None, num_classes=4)
            size = 32
        elif name == 'convnext':
            from torchvision.models.convnext import CNBlockConfig
            model = models.ConvNeXt([CNBlockConfig(8, 16, 1), CNBlockConfig(16, 32, 1),
                                    CNBlockConfig(32, 64, 1), CNBlockConfig(64, None, 1)], num_classes=4)
            size = 32
        else:
            model = models.SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[2, 2],
                num_heads=[2, 4], window_size=[2, 2], num_classes=4, stochastic_depth_prob=0)
            size = 32
        return model.eval(), (torch.randn(1, 3, size, size),)
    if name in ('t5', 'switch'):
        if name == 't5':
            from transformers import T5Config, T5ForConditionalGeneration
            model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=8,
                d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0,
                decoder_start_token_id=0, use_cache=False))
        else:
            from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
            config = SwitchTransformersConfig(vocab_size=32, d_model=16, d_kv=8, d_ff=32,
                num_layers=1, num_decoder_layers=1, num_heads=2, num_experts=2,
                expert_capacity=16, num_sparse_encoder_layers=1, num_sparse_decoder_layers=1,
                encoder_sparse_step=1, decoder_sparse_step=1, dropout_rate=0,
                decoder_start_token_id=0, use_cache=False)
            model = SwitchTransformersForConditionalGeneration(config)
        return Seq2SeqLogits(model).eval(), (torch.randint(1, 32, (2, 6)), torch.randint(1, 32, (2, 4)))
    if name == 'roformer':
        from transformers import RoFormerConfig, RoFormerModel
        config = RoFormerConfig(vocab_size=32, embedding_size=16, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
            return_dict=False)
        return RoFormerModel(config).eval(), (torch.randint(0, 32, (2, 6)),)
    if name == 'llama':
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, attention_dropout=0, use_cache=False, return_dict=False,
            attn_implementation='eager')
        return LlamaForCausalLM(config).eval(), (torch.randint(0, 32, (2, 6)),)
    if name == 'clip':
        from transformers import CLIPConfig, CLIPModel
        config = CLIPConfig(projection_dim=8, text_config=dict(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=16, eos_token_id=2, bos_token_id=1),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, image_size=32, patch_size=8))
        tokens = torch.randint(3, 32, (2, 6)); tokens[:, -1] = 2
        return CLIPLogits(CLIPModel(config)).eval(), (tokens, torch.randn(2, 3, 32, 32))
    if name == 'mamba':
        from transformers import MambaConfig, MambaModel
        config = MambaConfig(vocab_size=32, hidden_size=16, state_size=4,
            num_hidden_layers=1, expand=2, conv_kernel=4, time_step_rank=2,
            use_cache=False, return_dict=False)
        return MambaModel(config).eval(), (torch.randint(0, 32, (2, 6)),)
    if name == 'unet': return TinyUNet().eval(), (torch.randn(2, 3, 16, 16),)
    if name == 'vae': return TinyVAE().eval(), (torch.rand(2, 16), torch.randn(2, 4))
    if name == 'gan': return TinyGAN().eval(), (torch.randn(2, 4), torch.randn(2, 16))
    if name == 'ddpm': return TinyDDPM().eval(), (torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16), torch.tensor([2, 5]))
    adjacency = torch.tensor([[0.,1.,0.,0.,1.],[1.,0.,1.,0.,0.],[0.,1.,0.,1.,0.],[0.,0.,1.,0.,1.],[1.,0.,0.,1.,0.]])
    if name == 'gcn': return TinyGCN().eval(), (torch.randn(5, 8), adjacency)
    if name == 'gat': return TinyGAT().eval(), (torch.randn(5, 8), adjacency)
    raise ValueError('Unknown extended runtime implementation')
