"""Torch-dependent teaching models, imported only for runtime use."""
from torch import nn


class ThresholdPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return (self.linear(x) >= 0).to(x.dtype)


class TinyMixer(nn.Module):
    """Small structural example of Mixer; no attention or pretrained weights."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.token_norm = nn.LayerNorm(16)
        self.token_mixer = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 16))
        self.channel_norm = nn.LayerNorm(16)
        self.channel_mixer = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 4)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixer(self.channel_norm(x))
        return self.head(self.norm(x).mean(1))


class TinyMAE(nn.Module):
    """Fixed-mask MAE example: visible-only encoder, unshuffle, masked MSE.

    The fixed permutation makes inspection and checkpoint comparisons repeatable.
    Inputs include all image pixels, but masked pixel embeddings never enter the
    encoder. Positions are small fixed buffers, not the official sin-cos values.
    """
    def __init__(self):
        super().__init__()
        import torch
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.encoder = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.decoder_embed = nn.Linear(16, 8)
        self.decoder = nn.TransformerEncoderLayer(8, 2, 16, dropout=0., batch_first=True)
        self.head = nn.Linear(8, 48)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 8))
        order = torch.tensor([0, 5, 10, 15, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
        self.register_buffer('visible', order[:4])
        self.register_buffer('restore', order.argsort())
        self.register_buffer('masked', order[4:])
        self.register_buffer('encoder_position', torch.arange(256).reshape(1, 16, 16).float() / 256)
        self.register_buffer('decoder_position', torch.arange(128).reshape(1, 16, 8).float() / 128)

    def forward(self, images):
        import torch
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        visible = (tokens + self.encoder_position).index_select(1, self.visible)
        latent = self.decoder_embed(self.encoder(visible))
        shuffled = torch.cat((latent, self.mask_token.expand(images.shape[0], 12, -1)), dim=1)
        restored = shuffled.index_select(1, self.restore)
        return self.head(self.decoder(restored + self.decoder_position))

    def reconstruction_loss(self, images, predictions):
        # Same channel-last ordering as the official patchify/einsum path.
        targets = images.reshape(images.shape[0], 3, 4, 4, 4, 4)
        targets = targets.permute(0, 2, 4, 3, 5, 1).reshape(images.shape[0], 16, 48)
        return (predictions - targets).square().mean(-1).index_select(1, self.masked).mean()


class TinyDETR(nn.Module):
    """Small DETR-shaped detector with learned queries and parallel set output."""
    def __init__(self):
        super().__init__()
        import torch
        self.backbone = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.encoder = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.query_embed = nn.Embedding(5, 16)
        self.class_head = nn.Linear(16, 4)  # three classes plus no-object
        self.box_head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        self.register_buffer('position', torch.arange(256).reshape(1, 16, 16).float() / 256)

    def forward(self, images):
        tokens = self.backbone(images).flatten(2).transpose(1, 2) + self.position
        memory = self.encoder(tokens)
        queries = self.query_embed.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return __import__('torch').cat((self.class_head(decoded), self.box_head(decoded).sigmoid()), -1)

    def set_loss(self, predictions, target_classes, target_boxes):
        """Exact tiny bipartite assignment for two targets, then class/L1 loss."""
        import itertools
        import torch
        from torch.nn import functional as F
        assignments = list(itertools.permutations(range(predictions.shape[1]), 2))
        batch_losses, chosen_assignments = [], []
        for batch, (classes, boxes) in enumerate(zip(target_classes, target_boxes)):
            losses = []
            for first, second in assignments:
                chosen = predictions[batch, (first, second)]
                losses.append(F.cross_entropy(chosen[:, :4], classes) +
                    F.l1_loss(chosen[:, 4:], boxes))
            stacked = torch.stack(losses)
            index = int(stacked.detach().argmin())
            batch_losses.append(stacked[index])
            chosen_assignments.append(assignments[index])
        return torch.stack(batch_losses).mean(), chosen_assignments


class FourierPositionEncoder(nn.Module):
    """NeRF-style identity plus log-spaced sine/cosine features."""
    def __init__(self, frequencies):
        super().__init__()
        import torch
        self.register_buffer('frequencies', 2. ** torch.arange(frequencies).float())

    def forward(self, values):
        import torch
        encoded = [values]
        for frequency in self.frequencies:
            encoded.extend((torch.sin(values * frequency), torch.cos(values * frequency)))
        return torch.cat(encoded, -1)


class VolumeRenderer(nn.Module):
    """Alpha compositing along one ordered sample sequence per ray."""
    def forward(self, rgb, density, deltas):
        import torch
        alpha = 1. - torch.exp(-torch.relu(density.squeeze(-1)) * deltas)
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-6), 1), 1)[:, :-1]
        weights = alpha * transmittance
        return (weights.unsqueeze(-1) * rgb).sum(1)


class TinyNeRF(nn.Module):
    """Small NeRF-shaped field with view-dependent color and volume rendering."""
    def __init__(self):
        super().__init__()
        self.position_encoder = FourierPositionEncoder(2)
        self.direction_encoder = FourierPositionEncoder(1)
        self.point_mlp = nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.density_head = nn.Linear(32, 1)
        self.feature_head = nn.Linear(32, 16)
        self.view_head = nn.Sequential(nn.Linear(25, 16), nn.ReLU(), nn.Linear(16, 3), nn.Sigmoid())
        self.volume_renderer = VolumeRenderer()

    def forward(self, points, directions, deltas):
        features = self.point_mlp(self.position_encoder(points))
        density = self.density_head(features)
        view_input = __import__('torch').cat((self.feature_head(features), self.direction_encoder(directions)), -1)
        rgb = self.view_head(view_input)
        return self.volume_renderer(rgb, density, deltas)


class TinyPerceiver(nn.Module):
    """Input-length-independent latent bottleneck followed by latent processing."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(8, 16)
        self.latent_array = nn.Embedding(4, 16)
        self.cross_attention = nn.MultiheadAttention(16, 2, dropout=0., batch_first=True)
        self.latent_processor = nn.TransformerEncoderLayer(
            16, 2, 32, dropout=0., batch_first=True)
        self.classifier = nn.Linear(16, 4)

    def forward(self, inputs):
        projected = self.input_projection(inputs)
        latents = self.latent_array.weight.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        distilled, _ = self.cross_attention(latents, projected, projected, need_weights=False)
        processed = self.latent_processor(latents + distilled)
        return self.classifier(processed.mean(1))


class MeanNeighborAggregator(nn.Module):
    """Deterministic full-neighborhood mean used to expose GraphSAGE aggregation."""
    def forward(self, features, adjacency):
        degrees = adjacency.sum(-1, keepdim=True).clamp_min(1.)
        return adjacency @ features / degrees


class TinyGraphSAGE(nn.Module):
    """One GraphSAGE-mean layer with separate self and neighbor transforms."""
    def __init__(self):
        super().__init__()
        self.aggregator = MeanNeighborAggregator()
        self.self_projection = nn.Linear(8, 12)
        self.neighbor_projection = nn.Linear(8, 12)
        self.node_classifier = nn.Linear(12, 3)

    def forward(self, features, adjacency):
        neighbors = self.aggregator(features, adjacency)
        hidden = __import__('torch').relu(
            self.self_projection(features) + self.neighbor_projection(neighbors))
        return self.node_classifier(hidden)


class SumSetPool(nn.Module):
    """Permutation-invariant sum over the elements of a set."""
    def forward(self, features):
        return features.sum(1)


class SymmetricMaxPool(nn.Module):
    """PointNet's channel-wise symmetric maximum over unordered points."""
    def forward(self, features):
        return features.max(1).values


class TinyDeepSets(nn.Module):
    """Small rho(sum(phi(x))) set classifier with shared element encoding."""
    def __init__(self):
        super().__init__()
        self.element_encoder = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.set_pool = SumSetPool()
        self.set_head = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))

    def forward(self, elements):
        return self.set_head(self.set_pool(self.element_encoder(elements)))


class TinyPointNet(nn.Module):
    """Small PointNet classifier exposing shared point MLP and max pooling.

    This fixed example focuses on PointNet's core unordered-point path. The
    paper's optional input/feature transformation networks are catalogued but
    intentionally omitted from this compact runtime.
    """
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
        self.symmetric_pool = SymmetricMaxPool()
        self.shape_classifier = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))

    def forward(self, points):
        return self.shape_classifier(self.symmetric_pool(self.point_encoder(points)))


class TinyViewEncoder(nn.Module):
    """Small shared convolutional encoder used by two-view SSL examples."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, images):
        return self.features(images).flatten(1)


class TinySimCLR(nn.Module):
    """Shared encoder and nonlinear projection head for a positive view pair."""
    def __init__(self):
        super().__init__()
        self.encoder = TinyViewEncoder()
        self.projector = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, view_one, view_two):
        import torch
        first = torch.nn.functional.normalize(
            self.projector(self.encoder(view_one)), dim=-1)
        second = torch.nn.functional.normalize(
            self.projector(self.encoder(view_two)), dim=-1)
        return torch.stack((first, second), dim=1)

    def nt_xent_loss(self, projections, temperature=.2):
        """Symmetric in-batch normalized temperature-scaled cross entropy."""
        import torch
        count = projections.shape[0]
        flat = torch.cat((projections[:, 0], projections[:, 1]), dim=0)
        logits = flat @ flat.transpose(0, 1) / temperature
        logits = logits.masked_fill(torch.eye(2 * count, dtype=torch.bool,
            device=flat.device), -1e9)
        positives = torch.cat((torch.arange(count, 2 * count,
            device=flat.device), torch.arange(count, device=flat.device)))
        return torch.nn.functional.cross_entropy(logits, positives)


class TinyBYOL(nn.Module):
    """Online/target encoders with an online-only predictor and stop-gradient."""
    def __init__(self):
        super().__init__()
        import copy
        self.online_encoder = TinyViewEncoder()
        self.online_projector = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
        self.online_predictor = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for parameter in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            parameter.requires_grad_(False)

    def forward(self, view_one, view_two):
        import torch
        online_first = torch.nn.functional.normalize(self.online_predictor(
            self.online_projector(self.online_encoder(view_one))), dim=-1)
        online_second = torch.nn.functional.normalize(self.online_predictor(
            self.online_projector(self.online_encoder(view_two))), dim=-1)
        with torch.no_grad():
            target_first = torch.nn.functional.normalize(
                self.target_projector(self.target_encoder(view_one)), dim=-1)
            target_second = torch.nn.functional.normalize(
                self.target_projector(self.target_encoder(view_two)), dim=-1)
        return torch.stack((online_first, online_second, target_first,
                            target_second), dim=1)

    def bootstrap_loss(self, outputs):
        first = 2. - 2. * (outputs[:, 0] * outputs[:, 3].detach()).sum(-1)
        second = 2. - 2. * (outputs[:, 1] * outputs[:, 2].detach()).sum(-1)
        return (first + second).mean()

    @__import__('torch').no_grad()
    def update_target(self, momentum=.99):
        for online, target in zip(self.online_encoder.parameters(),
                                  self.target_encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1. - momentum)


class FixedNegativeQueue(nn.Module):
    """A registered, normalized MoCo dictionary used without mutation in tracing."""
    def __init__(self, size=16, width=8):
        super().__init__()
        import torch
        queue = torch.randn(size, width)
        self.register_buffer('entries', torch.nn.functional.normalize(queue, dim=-1))

    def forward(self):
        return self.entries


class TinyMoCo(nn.Module):
    """MoCo-shaped query/key encoders with EMA key path and a fixed queue."""
    def __init__(self):
        super().__init__()
        import copy
        self.query_encoder = TinyViewEncoder()
        self.query_projector = nn.Linear(16, 8)
        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.key_projector = copy.deepcopy(self.query_projector)
        self.negative_queue = FixedNegativeQueue()
        for parameter in list(self.key_encoder.parameters()) + list(self.key_projector.parameters()):
            parameter.requires_grad_(False)

    def forward(self, query_view, key_view):
        import torch
        query = torch.nn.functional.normalize(
            self.query_projector(self.query_encoder(query_view)), dim=-1)
        with torch.no_grad():
            key = torch.nn.functional.normalize(
                self.key_projector(self.key_encoder(key_view)), dim=-1)
        positive = (query * key).sum(-1, keepdim=True)
        negatives = query @ self.negative_queue().transpose(0, 1)
        return torch.cat((positive, negatives), dim=1)

    @__import__('torch').no_grad()
    def update_key_encoder(self, momentum=.99):
        pairs = zip(self.query_encoder.parameters(), self.key_encoder.parameters())
        for query, key in pairs:
            key.mul_(momentum).add_(query, alpha=1. - momentum)
        pairs = zip(self.query_projector.parameters(), self.key_projector.parameters())
        for query, key in pairs:
            key.mul_(momentum).add_(query, alpha=1. - momentum)


class TinyDINO(nn.Module):
    """Compact DINO student/EMA-teacher self-distillation path."""
    def __init__(self):
        super().__init__()
        import copy
        self.student_encoder = TinyViewEncoder()
        self.student_head = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 8))
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.register_buffer('center', __import__('torch').zeros(1, 8))
        for parameter in list(self.teacher_encoder.parameters()) + list(self.teacher_head.parameters()):
            parameter.requires_grad_(False)

    def forward(self, global_view, local_view):
        import torch
        student = self.student_head(self.student_encoder(local_view))
        with torch.no_grad():
            teacher = self.teacher_head(self.teacher_encoder(global_view)) - self.center
        return torch.stack((student, teacher), dim=1)

    def self_distillation_loss(self, outputs, student_temperature=.1, teacher_temperature=.04):
        import torch
        student_log = torch.nn.functional.log_softmax(outputs[:, 0] / student_temperature, -1)
        teacher_prob = torch.nn.functional.softmax(outputs[:, 1].detach() / teacher_temperature, -1)
        return -(teacher_prob * student_log).sum(-1).mean()


class GINLayer(nn.Module):
    """Sum aggregation followed by an MLP, with learnable epsilon."""
    def __init__(self, width=8):
        super().__init__()
        import torch
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(width, 16), nn.ReLU(), nn.Linear(16, width))

    def forward(self, features, adjacency):
        neighbors = adjacency @ features
        return self.mlp((1. + self.epsilon) * features + neighbors)


class GraphSumPool(nn.Module):
    def forward(self, features):
        return features.sum(1)


class TinyGIN(nn.Module):
    """One GIN layer, graph-level sum readout and classifier."""
    def __init__(self):
        super().__init__()
        self.gin_layer = GINLayer()
        self.graph_pool = GraphSumPool()
        self.classifier = nn.Linear(8, 3)

    def forward(self, features, adjacency):
        import torch
        hidden = torch.relu(self.gin_layer(features, adjacency))
        return self.classifier(self.graph_pool(hidden))


class SSDCore(nn.Module):
    """Small scalar-transition structured state-space dual recurrence."""
    def __init__(self, width=16, state_size=4):
        super().__init__()
        import torch
        self.state_input = nn.Linear(width, state_size)
        self.state_output = nn.Linear(state_size, width)
        self.log_decay = nn.Parameter(torch.zeros(state_size))

    def forward(self, sequence):
        import torch
        state = torch.zeros(sequence.shape[0], self.state_input.out_features,
                            device=sequence.device, dtype=sequence.dtype)
        decay = torch.sigmoid(self.log_decay)
        outputs = []
        for step in sequence.unbind(1):
            state = decay * state + self.state_input(step)
            outputs.append(self.state_output(state))
        return torch.stack(outputs, dim=1)


class TinyMamba2(nn.Module):
    """Compact Mamba-2/SSD block with input split, SSD core and gated output."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(8, 32)
        self.ssd_core = SSDCore(16, 4)
        self.output_gate = nn.SiLU()
        self.output_projection = nn.Linear(16, 8)

    def forward(self, sequence):
        projected, gate = self.input_projection(sequence).chunk(2, dim=-1)
        mixed = self.ssd_core(projected)
        return self.output_projection(mixed * self.output_gate(gate))


class SqueezeExcitation(nn.Module):
    """Global squeeze followed by a bottleneck channel gate."""
    def __init__(self, channels=8, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, features):
        scale = self.excitation(self.squeeze(features).flatten(1))
        return features * scale[:, :, None, None]


class SEResidualBlock(nn.Module):
    """Two convolutions with SE recalibration before residual addition."""
    def __init__(self, channels=8):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1))
        self.se = SqueezeExcitation(channels)

    def forward(self, features):
        return __import__('torch').relu(features + self.se(self.convolutions(features)))


class TinySENet(nn.Module):
    """Compact SE-ResNet-shaped image classifier."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.block = SEResidualBlock(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, images):
        features = self.block(self.stem(images))
        return self.head(self.pool(features).flatten(1))


class DiagonalS4Kernel(nn.Module):
    """Small stable diagonal SSM recurrence in the S4D family."""
    def __init__(self, width=16, state_size=4):
        super().__init__()
        import torch
        self.input_map = nn.Linear(width, state_size)
        self.output_map = nn.Linear(state_size, width)
        self.log_step = nn.Parameter(torch.full((state_size,), -2.))
        self.log_decay = nn.Parameter(torch.linspace(-1., -3., state_size))

    def forward(self, sequence):
        import torch
        step = torch.nn.functional.softplus(self.log_step)
        decay = torch.exp(-step * torch.exp(self.log_decay))
        state = torch.zeros(sequence.shape[0], self.input_map.out_features,
                            device=sequence.device, dtype=sequence.dtype)
        outputs = []
        for token in sequence.unbind(1):
            state = decay * state + step * self.input_map(token)
            outputs.append(self.output_map(state))
        return torch.stack(outputs, dim=1)


class TinyS4(nn.Module):
    """Compact residual S4D-style long-sequence classifier."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(8, 16)
        self.state_kernel = DiagonalS4Kernel(16, 4)
        self.normalization = nn.LayerNorm(16)
        self.head = nn.Linear(16, 4)

    def forward(self, sequence):
        projected = self.input_projection(sequence)
        mixed = self.normalization(projected + self.state_kernel(projected))
        return self.head(mixed.mean(1))


class MultiScaleRetention(nn.Module):
    """Parallel causal retention with one fixed decay per head."""
    def __init__(self, width=16, heads=2):
        super().__init__()
        import torch
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.register_buffer('decays', torch.tensor([.8, .95]))

    def forward(self, sequence):
        import torch
        batch, length, width = sequence.shape
        q, k, value = self.qkv(sequence).chunk(3, dim=-1)
        def split(tensor):
            return tensor.reshape(batch, length, self.heads,
                                  self.head_width).transpose(1, 2)
        q, k, value = split(q), split(k), split(value)
        positions = torch.arange(length, device=sequence.device)
        distance = positions[:, None] - positions[None, :]
        causal = distance >= 0
        decay = self.decays[:, None, None].pow(distance.clamp_min(0))
        weights = (q @ k.transpose(-1, -2)) / self.head_width ** .5
        weights = weights * decay[None] * causal[None, None]
        retained = weights @ value
        retained = retained.transpose(1, 2).reshape(batch, length, width)
        return self.output(retained)


class TinyRetNet(nn.Module):
    """Compact gated multi-scale retention block."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(8, 16)
        self.retention = MultiScaleRetention(16, 2)
        self.output_gate = nn.Sequential(nn.Linear(16, 16), nn.SiLU())
        self.output_projection = nn.Linear(16, 8)

    def forward(self, sequence):
        projected = self.input_projection(sequence)
        retained = self.retention(projected)
        return self.output_projection(retained * self.output_gate(projected))


class ODEVectorField(nn.Module):
    """Autonomous neural vector field used by the fixed ODE solver."""
    def __init__(self, width=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))

    def forward(self, state):
        return self.network(state)


class FixedEulerSolver(nn.Module):
    """Four explicit Euler steps, kept as a module boundary for inspection."""
    def __init__(self, vector_field, steps=4):
        super().__init__()
        self.vector_field = vector_field
        self.steps = steps

    def forward(self, initial_state):
        state = initial_state
        step_size = 1. / self.steps
        for _ in range(self.steps):
            state = state + step_size * self.vector_field(state)
        return state


class TinyNeuralODE(nn.Module):
    """Compact continuous-depth classifier with a differentiable ODE block."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(6, 12)
        self.vector_field = ODEVectorField(12)
        self.ode_solver = FixedEulerSolver(self.vector_field, 4)
        self.classifier = nn.Linear(12, 3)

    def forward(self, features):
        state = self.input_projection(features)
        return self.classifier(self.ode_solver(state))


class SpectralConvolution1d(nn.Module):
    """Low-frequency Fourier multiplier for a real one-dimensional field."""
    def __init__(self, width=8, modes=6):
        super().__init__()
        import torch
        self.modes = modes
        scale = 1. / width
        self.weight = nn.Parameter(scale * torch.randn(width, width, modes,
                                                       dtype=torch.cfloat))

    def forward(self, field):
        import torch
        spectrum = torch.fft.rfft(field, dim=-1)
        truncated = spectrum[:, :, :self.modes]
        mixed = torch.einsum('bim,iom->bom', truncated, self.weight)
        output = torch.zeros(field.shape[0], field.shape[1], spectrum.shape[-1],
                             dtype=spectrum.dtype, device=field.device)
        output[:, :, :self.modes] = mixed
        return torch.fft.irfft(output, n=field.shape[-1], dim=-1)


class TinyFNO(nn.Module):
    """One-dimensional Fourier neural operator with spectral and local paths."""
    def __init__(self):
        super().__init__()
        self.lifting = nn.Conv1d(2, 8, 1)
        self.spectral_convolution = SpectralConvolution1d(8, 6)
        self.local_convolution = nn.Conv1d(8, 8, 1)
        self.projection = nn.Sequential(
            nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))

    def forward(self, field):
        import torch
        grid = torch.linspace(0, 1, field.shape[-1], device=field.device,
                              dtype=field.dtype).expand(field.shape[0], 1, -1)
        lifted = self.lifting(torch.cat((field, grid), dim=1))
        hidden = torch.nn.functional.gelu(
            self.spectral_convolution(lifted) + self.local_convolution(lifted))
        return self.projection(hidden.transpose(1, 2)).transpose(1, 2)


class RWKVTimeMix(nn.Module):
    """Small recurrent weighted key-value time mixer inspired by RWKV-4."""
    def __init__(self, width=12):
        super().__init__()
        import torch
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.receptance = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.time_decay = nn.Parameter(torch.zeros(width))

    def forward(self, sequence):
        import torch
        numerator = torch.zeros_like(sequence[:, 0])
        denominator = torch.zeros_like(sequence[:, 0])
        decay = torch.sigmoid(self.time_decay)
        outputs = []
        for token in sequence.unbind(1):
            key_weight = torch.exp(torch.clamp(self.key(token), -8., 8.))
            numerator = decay * numerator + key_weight * self.value(token)
            denominator = decay * denominator + key_weight
            mixed = numerator / denominator.clamp_min(1e-6)
            outputs.append(self.output(torch.sigmoid(self.receptance(token)) * mixed))
        return torch.stack(outputs, dim=1)


class RWKVChannelMix(nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.key = nn.Linear(width, 2 * width)
        self.value = nn.Linear(2 * width, width)
        self.receptance = nn.Linear(width, width)

    def forward(self, sequence):
        import torch
        hidden = torch.relu(self.key(sequence)).square()
        return torch.sigmoid(self.receptance(sequence)) * self.value(hidden)


class TinyRWKV(nn.Module):
    """Compact attention-free token model with RWKV time/channel mixing."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.time_mix = RWKVTimeMix(12)
        self.channel_mix = RWKVChannelMix(12)
        self.normalization = nn.LayerNorm(12)
        self.lm_head = nn.Linear(12, 32)

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        hidden = hidden + self.time_mix(hidden)
        hidden = self.normalization(hidden + self.channel_mix(hidden))
        return self.lm_head(hidden)


class TinyLatentDiffusion(nn.Module):
    """Compact LDM path: first-stage compression, conditioned latent denoising.

    The fixed example exposes the architectural boundary between the image
    autoencoder and diffusion model. It predicts one latent noise field; it is
    not a multi-step sampler or a reproduction of a trained text-to-image LDM.
    """
    def __init__(self):
        super().__init__()
        import torch
        from .extended_models import TinyUNet
        self.first_stage_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1))
        self.first_stage_decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1))
        self.condition_encoder = nn.Sequential(
            nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
        self.time_embedding = nn.Sequential(
            nn.Embedding(10, 8), nn.Linear(8, 8), nn.SiLU())
        self.latent_denoiser = TinyUNet(8, 8)
        self.register_buffer('alpha_bar',
            torch.cumprod(1 - torch.linspace(.0001, .02, 10), 0))

    def forward(self, images, condition_tokens, noise, timestep):
        latent = self.first_stage_encoder(images)
        alpha = self.alpha_bar[timestep].reshape(-1, 1, 1, 1)
        noisy = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        condition = self.condition_encoder(condition_tokens).mean(1)
        context = condition + self.time_embedding(timestep)
        predicted_noise = self.latent_denoiser(
            noisy + context.unsqueeze(-1).unsqueeze(-1))
        return predicted_noise, self.first_stage_decoder(predicted_noise)

    @staticmethod
    def denoising_loss(predicted_noise, noise):
        return (predicted_noise - noise).square().mean()


class AdaLNDiTBlock(nn.Module):
    """DiT transformer block with timestep/class adaptive normalization."""
    def __init__(self, width=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, 2, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, tokens, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(condition).chunk(6, dim=-1))
        attention_input = self.norm1(tokens) * (1 + scale_a[:, None]) + shift_a[:, None]
        attended, _ = self.attention(attention_input, attention_input,
                                     attention_input, need_weights=False)
        tokens = tokens + gate_a[:, None] * attended
        mlp_input = self.norm2(tokens) * (1 + scale_m[:, None]) + shift_m[:, None]
        return tokens + gate_m[:, None] * self.mlp(mlp_input)


class TinyDiT(nn.Module):
    """Small class-conditional latent Diffusion Transformer noise predictor."""
    def __init__(self):
        super().__init__()
        import torch
        self.patch_embed = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.time_embed = nn.Sequential(
            nn.Embedding(10, 16), nn.Linear(16, 16), nn.SiLU())
        self.label_embed = nn.Embedding(5, 16)
        self.blocks = nn.ModuleList([AdaLNDiTBlock(16), AdaLNDiTBlock(16)])
        self.final_norm = nn.LayerNorm(16)
        self.final_layer = nn.Linear(16, 16)
        self.register_buffer('position',
            torch.arange(256).reshape(1, 16, 16).float() / 256)

    def forward(self, latent, timestep, label):
        tokens = self.patch_embed(latent).flatten(2).transpose(1, 2) + self.position
        condition = self.time_embed(timestep) + self.label_embed(label)
        for block in self.blocks:
            tokens = block(tokens, condition)
        patches = self.final_layer(self.final_norm(tokens))
        batch = latent.shape[0]
        # Inverse of 2x2 patchification: (B, 16 tokens, 4*2*2) -> (B, 4, 8, 8).
        return patches.reshape(batch, 4, 4, 4, 2, 2).permute(
            0, 1, 2, 4, 3, 5).reshape(batch, 4, 8, 8)
        for online, target in zip(self.online_projector.parameters(),
                                  self.target_projector.parameters()):
            target.mul_(momentum).add_(online, alpha=1. - momentum)
