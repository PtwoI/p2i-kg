# DETR and NeRF extension

This additive fragment contributes 25 nodes and 57 sourced relationships. It
was checked against the primary papers and the authors' official releases:

- DETR: [paper](https://arxiv.org/abs/2005.12872),
  [`facebookresearch/detr`](https://github.com/facebookresearch/detr). Claims
  point to paper §3 and to `models/detr.py` / `models/transformer.py` for the
  learned queries, encoder-decoder, class and box heads, and `SetCriterion`.
- NeRF: [paper](https://arxiv.org/abs/2003.08934),
  [`bmild/nerf`](https://github.com/bmild/nerf). Claims point to paper §4–5 and
  to `run_nerf_helpers.py` / `run_nerf.py` for `Embedder`,
  `init_nerf_model`, `run_network`, and `raw2outputs`.

The runtimes are deliberately small structural examples with random weights.
TinyDETR uses a 16×16 input, five learned object queries, one encoder and one
decoder layer, three object classes plus no-object, and class/box heads. Its
teaching loss performs an exact exhaustive bipartite assignment for two targets;
it does not reproduce COCO augmentation, auxiliary losses, GIoU, or the official
ResNet-50 configuration.

TinyNeRF evaluates six ordered samples on each of two rays. It includes
log-spaced sine/cosine coordinate features, a point MLP, separate density and
view-conditioned RGB paths, and differentiable alpha compositing. It omits real
camera ray generation, stratified and hierarchical sampling, the paper's full
network width/depth, and coarse/fine networks.

Both factories support P2I runtime tracing, module inspection, committed
parameter removal, checkpoint restoration into a fresh matching template, and
backward validation. A downloaded checkpoint for either custom forward requires
that template explicitly:

```python
import p2i
from p2i_kg.runtime import make_model

model, inputs = make_model("detr")  # or "nerf"
harness = p2i.Harness.load("example.p2i", model=model)
```

`docs/detr-nerf.json` is an idempotent fragment for `p2i-kg extend`; it does not
replace existing nodes, claims, titles, or other user curation.
