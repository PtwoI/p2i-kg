# P2I Architecture Atlas

A local educational knowledge graph connecting neural models, building-block
concepts and representative papers, with live P2I model exploration.

The initial catalog contains **8 models, 12 concepts, 8 papers and 41 edges**:
Perceptron, MLP, CNN/LeNet family, Elman-style RNN, LSTM, GRU, ResNet and
Transformer. Years identify representative references, not universal invention
dates. Every edge records evidence and whether its interpretation is documented,
structural or pedagogical. A single straight historical lineage is not assumed.

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

The Perceptron example uses a real threshold and is not differentiable. Other
bundled models are small educational examples, not full reproductions of every
paper. In particular TinyTransformer illustrates Attention/FFN blocks without
position encoding or a full encoder-decoder. Model cards describe these scopes.

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
p2i-kg export graph.json
p2i-kg export graph.jsonld --format jsonld
p2i-kg import graph.json
```

`--database PATH` before the command selects a different database. Imports
validate IDs, relationship types, domain/range, references and paper metadata
before the atomic SQLite write. Identical imports retain their content hash.
Edit `src/p2i_kg/catalog.json` and run validation/tests to curate the next catalog.
An existing database changes only on an explicit import; upgrading the package
does not overwrite locally curated content.

Relations are `uses` (model→concept), `documented_in` (model→paper), `extends`
and `alternative_to` (model→model). JSON-LD includes both the relationships and
reified claims retaining explanation, evidence and interpretation. Comparison
shows currently curated concepts, not performance rankings or automatic claims
of historical influence. This release seeds a maintained ontology; automated
literature discovery and real-time updating are subsequent work.

## Validation

```bash
python -m pytest -q
```

Core tests run without P2I. Runtime tests additionally exercise all eight model
factories when compatible P2I/TorchVision packages are installed. The browser
acceptance script uses the Playwright installation from `p2i-gui/frontend`:

```bash
p2i-kg --database /tmp/atlas.sqlite3 serve --port 8010
P2I_KG_URL=http://127.0.0.1:8010 node tests/browser.mjs
```

CI separately checks the graph-only installation, P2I runtime integration and
the interactive explorer. This repository extends P2I rather than replacing its
core, GUI, SDK or CLI packages.
