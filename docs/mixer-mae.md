# MLP-Mixer and masked autoencoding extension

This additive fragment contributes 2 models, 4 concepts, 5 components, 1 objective,
1 task and 2 papers: **15 nodes and 34 claims**. The resulting catalog has
32 models, 154 nodes and 321 claims. Existing records and stable claim IDs are
unchanged. To extend an existing curated database explicitly:

```sh
p2i-kg extend docs/mixer-mae.json
```

Repeated imports are idempotent. `upgrade` also preserves local edits; package
installation itself never replaces an existing database.

## Primary sources and official implementations

- MLP-Mixer, [original paper](https://arxiv.org/abs/2105.01601), §2/Fig.1;
  [official Google JAX implementation](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py),
  `MixerBlock` and `MlpMixer`: patch projection, alternating token/channel MLPs,
  pre-normalization and residual connections. `alternative_to` ViT describes a
  structural alternative, not historical ancestry or feature inheritance.
- MAE, [original paper](https://arxiv.org/abs/2111.06377), §3/Fig.1;
  [official Meta PyTorch implementation](https://github.com/facebookresearch/mae/blob/main/models_mae.py),
  `random_masking`, `forward_encoder`, `forward_decoder`, `forward_loss`:
  visible-only encoder, mask-token insertion, inverse permutation, decoder and
  masked-only pixel MSE. Reference year 2021 is the first preprint year (CVPR 2022).
  MAE extends the ViT encoder design; it does not use encoder-decoder cross-attention.

## Executable scope

`make_model('mixer')`: 16×16 RGB image, 16 patches, width 16, one Mixer block,
token MLP 16→8→16, channel MLP 16→32→16, global mean and 4-class head.
Bindings expose `patch_embed`, `token_mixer` and `channel_mixer`.

`make_model('mae')`: same input, 4 visible of 16 patches, width-16 encoder,
width-8 decoder, one self-attention layer each, 48-pixel patch head.
Bindings expose patch embedding, encoder, decoder and pixel predictor. The fixed
75% mask and fixed position buffers permit exact replay; CLS token, official
sin-cos positions, random mask resampling, pixel normalization and training recipe
are omitted. This is a small structural example with randomly initialized weights,
not an official-model reproduction or a benchmark. `reconstruction_loss` computes
channel-last patch targets and masked-only MSE.

Both factories reuse supported Linear, Conv2d, LayerNorm, GELU and Transformer
modules. No core adapters or dependency changes are needed. Custom forward code
is reconstructed with an explicit fresh factory template:

```python
import p2i
from p2i_kg.runtime import make_model
model, inputs = make_model('mae')
harness = p2i.Harness.load('mae.p2i', model=model)
```

Tests check observed operations, all 7 actual module bindings, committed edits,
checkpoint restoration into a separately instantiated template, exact edited-output
agreement and backward validation. Behavioral tests verify that only 4 tokens enter
the MAE encoder, changing masked input pixels cannot change predictions, visible
prediction errors do not affect masked loss, and reconstruction gradients reach
both mask token and patch projection. Curation preservation and fragment
idempotence are tested against a pre-extension catalog with user-edited text.
