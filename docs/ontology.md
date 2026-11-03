# Neural architecture ontology

The vocabulary is generated from the same relation definitions as the API and
catalog validator. Stable IRIs use `https://github.com/PtwoI/p2i-kg#` with paths
such as `model/bert` and `component/encoder`. Claims have stable IDs derived from
the source/relation/target tuple; adding an unrelated claim does not renumber them.

| Domain | Relation | Range |
| --- | --- | --- |
| Model / Component | uses | Concept |
| Model / Component | has_component | Component |
| Concept | subconcept_of | Concept |
| Model | trained_with | Objective |
| Model | applied_to | Task |
| Model | extends / alternative_to | Model |
| Non-paper node | documented_in | Paper |

Models and concepts are individuals of vocabulary classes. A structural lineage
link is not an `rdfs:subClassOf` axiom. `subconcept_of` is declared transitive and
`alternative_to` symmetric in OWL. Multi-kind domains use `owl:unionOf`; multiple
independent `rdfs:domain` axioms would imply an intersection instead.

OWL domain/range axioms support inference. They are not database validation
constraints. Generated SHACL Shapes independently validate labels, summaries,
paper metadata, claim evidence, and relationship domains/ranges. The Python
validator additionally rejects duplicate IDs, dangling links and cycles in
composition or concept taxonomy. This is a bounded curated ontology, not an
implementation of a complete OWL DL reasoner.

Standards: [OWL 2 Primer](https://www.w3.org/TR/owl-primer/) and
[SHACL](https://www.w3.org/TR/shacl/). RDF data and vocabulary can be loaded into
standard RDF tooling; JSON-LD preserves the same RDF triples, including reference
year, authors, runtime scope, evidence and claim interpretation.

## Questions the graph answers

- Which models use attention, including window/graph/self/cross attention?
- Which architectures use position information and an autoregressive objective?
- What components, objectives and tasks distinguish BERT from GPT-2?
- Which actual P2I module implements the decoder cross-attention in the example?

`GET /api/kg/query?concept=concept:attention` returns model matches and ordered
supporting `proof_edges`. Composition followed by `uses` and concept hierarchy
links is the entire inference rule. A historical `extends` edge never causes
features to be inherited. In particular Mamba is an alternative to Transformer,
so the graph does not infer that Mamba uses self-attention.

`GET /api/kg/neighborhood/model:transformer?hops=2&limit=40` bounds rendering.
`GET /api/kg/inspect/model:transformer` returns a real P2I ArchitectureIR and
verified component/module path bindings for the current fixed example. Bindings
are factory-specific and checked against actual modules, with current dimensions
and editability. They are not guesses about arbitrary models. Conceptual model
cards and executable examples retain separate scopes.

## Incremental extension

A fragment can add nodes and claims using already-existing paper evidence:

```json
{
  "schema_version": "0.2",
  "nodes": [{"id":"concept:teaching_topic", "kind":"concept",
    "label":"Teaching topic", "summary":"A curated teaching concept."}],
  "edges": [{"source":"model:transformer", "relation":"uses",
    "target":"concept:teaching_topic", "basis":"pedagogical",
    "explanation":"A teaching relationship, not a paper discovery claim.",
    "evidence":["paper:transformer"]}]
}
```

Use `p2i-kg extend fragment.json`. The HTTP endpoint expects
`{"expected_revision":"<current /api/kg revision>","patch":{...}}`.
The database serializes extensions in one transaction. A concurrent extension
using an old revision fails rather than overwriting another contributor's work.
The title stays unchanged. Conflicting existing records require an intentional
full-catalog edit/import. `upgrade` alone keeps existing conflicts and reports
which entries need reconciliation.

Continue expanding by architectural differences: verify a primary paper, add
its metadata, connect the model to precise concepts/components/objectives/tasks,
then add a runtime where it improves understanding. If P2I cannot represent a
necessary module, add a trusted reconstruction adapter and exercise tracing,
editing, independent checkpoint restoration and backward before binding it.
Useful future branches should continue to add architectural differences rather
than near-duplicate scale variants, while preserving executable coverage.

The bundled Perceiver example fixes four latent queries while allowing the input
length to vary; its input projection, latent array, cross-attention, latent
processor and classifier are real P2I-bound modules. The GraphSAGE-mean example
uses deterministic full-neighborhood means so structural comparisons are
repeatable, while the catalog records the original sampled-neighborhood design.
Its aggregator, self/neighbor projections and node classifier are also bound to
the traced module paths.

The bundled Deep Sets and PointNet examples make their contrasting symmetric
reductions executable: Deep Sets applies sum pooling after a shared element
encoder, whereas PointNet applies channel-wise max pooling after a shared point
MLP. Both examples are permutation invariant and expose the encoder, reduction
and readout as verified P2I module bindings.

The bundled SimCLR and BYOL examples expose complementary self-supervised paths.
SimCLR shares one encoder/projector across two views and provides an in-batch
NT-Xent loss. BYOL separates online and stop-gradient target paths, adds an
online-only predictor and provides an explicit EMA target update. Augmentation
is outside both fixed runtime graphs; their runtime notes therefore state that
the two input tensors are already-created views.

The bundled MoCo and DINO examples add two distinct momentum-teacher families:
MoCo contrasts a query against a fixed queue-backed dictionary, while DINO
matches a student distribution to a centered, stop-gradient teacher. The GIN
example exposes learnable-epsilon sum aggregation and graph-level sum readout.
The Mamba-2 example isolates the input split, a small SSD recurrence, output
gate and projection; it documents, but does not pretend to reproduce, the
official CUDA chunk-scan kernel.

The SENet example binds the residual block, global squeeze and excitation MLP
that produce input-dependent channel scales. The S4 example exposes a compact
diagonal S4D-style state kernel, residual normalization and projections while
the catalog retains the original S4 DPLR/Cauchy-kernel contribution. The RetNet
example executes parallel causal retention with two decay scales and binds its
QKV projection, retention core, gate and output projections.

The bundled LDM and DiT examples connect two successive diffusion changes. LDM
separates first-stage image compression from conditioned latent denoising; DiT
keeps that latent diffusion setting while replacing the U-Net with latent patch
tokens and adaLN-conditioned Transformer blocks. Both execute one deterministic
noise-prediction step with random weights. Their runtime notes explicitly exclude
large pretrained encoders, complete iterative sampling and production-scale
training so the executable graph is not confused with a full generator.
