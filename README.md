# P2I Architecture Atlas

A local educational knowledge graph connecting neural models, building-block
concepts and representative papers, with live P2I model exploration.

Version 0.2 contains **38 models, 222 nodes and 478 sourced relationships**.
Alongside the original Perceptron/MLP/CNN/RNN/LSTM/GRU/ResNet/Transformer path,
it includes VGG, Inception, DenseNet, MobileNet, EfficientNet, ConvNeXt, U-Net,
BERT, GPT-2, T5, ViT, Swin, RoFormer, LLaMA (2023), VAE, GAN, DDPM, GCN, GAT,
CLIP, Switch Transformer, Mamba, MLP-Mixer, MAE, DETR, NeRF, Perceiver,
GraphSAGE, Deep Sets and PointNet. Reference years
identify first-public-paper years, not universal invention dates. Each claim
retains its evidence and its documented, structural or pedagogical interpretation.

The ontology distinguishes **models, concepts, components, training objectives,
tasks and papers**. Explore one/two-hop neighborhoods, compare models across
these facets, follow concept hierarchies, and inspect the links between curated
components and real P2I modules. OWL/RDF Turtle, JSON-LD and SHACL exports are
available. See [the ontology guide](docs/ontology.md).

## Run the graph

```bash
git clone https://github.com/PtwoI/p2i-kg.git
cd p2i-kg
python -m pip install -e '.[test]'
p2i-kg serve --port 8010
```

Open http://127.0.0.1:8010. Search models, concepts and papers, inspect sourced
relationships, filter model families, and compare curated concept sets. Catalog
state is stored in `~/.p2i/architecture-kg.sqlite3`. No account, model download or
external LLM is required. Graph-only use does not import PyTorch.

## Open an executable architecture

Install matching P2I source checkouts first (these commands do not assume PyPI
publication). From a directory containing the repositories:

```bash
git clone https://github.com/PtwoI/p2i-core.git
git clone https://github.com/PtwoI/p2i-gui.git
python -m pip install -e ./p2i-core -e ./p2i-gui -e './p2i-kg[runtime]'
p2i-kg open lstm --port 8000
```

The web explorer's **P2I로 열기** button starts a loopback session and shows a
link to its model explorer. At most three sessions run at once; close sessions
from their model card. They stop when the KG server shuts down. ResNet uses
TorchVision's randomly initialized ResNet18 and requires a TorchVision build
compatible with your PyTorch installation. No pretrained weights are fetched.

BERT/GPT-2 use small actual Hugging Face configurations (`pip install -e
'.[language]'`); ViT uses a small TorchVision configuration. No model weights
are downloaded. Only cards marked with a runtime have executable examples.

Deep Sets and PointNet use compact, dependency-free teaching implementations.
They expose shared per-element/point encoders and their distinct symmetric
reductions (sum versus max) as actual P2I module bindings. The PointNet runtime
focuses on its core classification path and omits the paper's optional T-Nets
and segmentation branch; those structures remain represented in the ontology.

The Perceptron example uses a real threshold and is not differentiable. Other
bundled models are small educational examples, not full reproductions of every
paper. In particular TinyTransformer illustrates Attention/FFN blocks without
position encoding or a full encoder-decoder. The Transformer catalog card now
uses P2I's separate `seq2seq` example, with sinusoidal positions, encoder memory,
causal target masking and decoder cross-attention. It is teacher-forced and
randomly initialized. Model cards describe these scopes.

Checkpoint downloads can reopen in their active P2I session. Bundled P2I demos
also load independently. For Perceptron and external models, explicitly provide
the matching Python template:

```python
import p2i
from p2i_kg.runtime import make_model
model, inputs = make_model('perceptron')
harness = p2i.Harness.load('perceptron.p2i', model=model)
```

## Maintain and export the KG

```bash
p2i-kg validate
p2i-kg validate --rdf  # install .[test] for the SHACL validator
p2i-kg export graph.json
p2i-kg export graph.jsonld --format jsonld
p2i-kg import graph.json
p2i-kg export graph.ttl --format turtle
p2i-kg export ontology.ttl --format ontology
p2i-kg export shapes.ttl --format shapes
p2i-kg query --concept concept:attention
p2i-kg query --concept concept:position --objective objective:autoregressive
p2i-kg extend fragment.json
p2i-kg upgrade  # add bundled catalog entries while retaining local curation
```

`--database PATH` before the command selects a different database. Imports
validate IDs, relationship types, domain/range, references and paper metadata
before the atomic SQLite write. Identical imports retain their content hash.
Edit `src/p2i_kg/catalog.json` and run validation/tests to curate the next catalog.
An existing database changes only on an explicit import; upgrading the package
does not overwrite locally curated content.

Relationships are `uses`, `has_component`, `subconcept_of`, `trained_with`,
`applied_to`, `documented_in`, `extends` and `alternative_to`. Model nodes are
OWL individuals describing architectures; `extends` does not declare an OWL
subclass or automatically inherit features. Concept queries traverse composition
and explicit concept-broader relations, returning supporting claim paths.

`extend` appends a validated fragment atomically and rejects conflicting IDs or
stale revisions. Repeating an identical fragment is idempotent. `upgrade` adds
new bundled entries to an existing database while keeping edited local nodes
and claims, reporting retained conflicts. Nothing silently overwrites a local
catalog on package upgrade. The web UI also accepts JSON fragments.
Schema 0.1 catalogs remain readable and normalize to schema 0.2 on explicit save.
See [extending the ontology](docs/ontology.md#incremental-extension).

## Validation

```bash
python -m pytest -q
```

Core tests run without P2I. Runtime tests additionally exercise all bundled model
factories when compatible P2I/TorchVision packages are installed. The browser
acceptance script uses the Playwright installation from `p2i-gui/frontend`:

```bash
p2i-kg --database /tmp/atlas.sqlite3 serve --port 8010
P2I_KG_URL=http://127.0.0.1:8010 node tests/browser.mjs
```

CI separately checks the graph-only installation, P2I runtime integration and
the interactive explorer. This repository extends P2I rather than replacing its
core, GUI, SDK or CLI packages.

## Map viewer

The WebUI opens directly on an interactive architecture map. Model nodes are arranged by representative-paper year, with family colors. Select a node for its concepts, components, changes and papers; select an edge for its meaning and evidence. Expand a node to see the surrounding ontology, including concept-to-model links. Drag or use touch gestures to pan and zoom. Comparison and the node list open over the map. There are no course screens, progress tracking, or notes.

```sh
p2i-kg serve                       # local catalog + P2I execution
p2i-kg viewer ./viewer             # portable read-only export
p2i-kg viewer ./viewer --runtime   # also trace installed runtime examples
```

The static export uses relative URLs and can be served at a subdirectory such as `/apps/p2imap/`. Its catalog, RDF exports and optional module snapshots work without Python. Execution, graph extension and model editing use the local server. The manifest records the catalog revision and available snapshots.
