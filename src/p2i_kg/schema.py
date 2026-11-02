"""Small typed ontology; relationship evidence and interpretation stay explicit."""
from typing import Literal
from hashlib import sha256
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


KINDS = ('model', 'concept', 'paper', 'component', 'objective', 'task')
RELATIONS = {
    'uses': (('model', 'component'), ('concept',)),
    'documented_in': (('model', 'concept', 'component', 'objective', 'task'), ('paper',)),
    'extends': (('model',), ('model',)),
    'alternative_to': (('model',), ('model',)),
    'has_component': (('model', 'component'), ('component',)),
    'subconcept_of': (('concept',), ('concept',)),
    'trained_with': (('model',), ('objective',)),
    'applied_to': (('model',), ('task',)),
}


class Node(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-z0-9:_-]+$')
    kind: Literal['model', 'concept', 'paper', 'component', 'objective', 'task']
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    reference_year: int | None = Field(default=None, ge=1900, le=2100)
    family: str | None = None
    url: HttpUrl | None = None
    authors: list[str] = Field(default_factory=list)
    runtime_id: str | None = None
    runtime_note: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str
    relation: Literal['uses', 'documented_in', 'extends', 'alternative_to', 'has_component', 'subconcept_of', 'trained_with', 'applied_to']
    target: str
    basis: Literal['documented', 'structural', 'pedagogical']
    explanation: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    locator: str | None = None

    @property
    def claim_id(self):
        key = '\0'.join((self.source, self.relation, self.target))
        return 'claim:' + sha256(key.encode()).hexdigest()[:24]


class KnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['0.2'] = '0.2'

    @model_validator(mode='before')
    @classmethod
    def migrate(cls, data):
        if isinstance(data, dict) and data.get('schema_version') == '0.1':
            return {**data, 'schema_version': '0.2'}
        return data
    title: str = 'P2I Architecture Atlas'
    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode='after')
    def validate_graph(self):
        nodes = {node.id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError('Duplicate node ID')
        signatures = set()
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError('Dangling relationship')
            if edge.source == edge.target:
                raise ValueError('Self relationships are not supported')
            signature = (edge.source, edge.relation, edge.target)
            if signature in signatures:
                raise ValueError('Duplicate relationship')
            signatures.add(signature)
            domain, range_ = RELATIONS[edge.relation]
            if nodes[edge.source].kind not in domain or nodes[edge.target].kind not in range_:
                raise ValueError('Relationship domain/range mismatch')
            if any(key not in nodes or nodes[key].kind != 'paper' for key in edge.evidence):
                raise ValueError('Relationship evidence must reference paper nodes')
        for node in self.nodes:
            if node.kind == 'paper' and (node.url is None or not node.authors or node.reference_year is None):
                raise ValueError('Paper nodes require URL, authors and reference year')
            if node.kind != 'model' and node.runtime_id:
                raise ValueError('Only models can have runtime implementations')
        # Taxonomy and composition are DAGs; historical relationships are not
        # interpreted as class inheritance or used to inherit model properties.
        for relation in ('subconcept_of', 'has_component'):
            adjacency = {}
            for edge in self.edges:
                if edge.relation == relation:
                    adjacency.setdefault(edge.source, []).append(edge.target)
            done, active = set(), set()
            def visit(key):
                if key in active:
                    raise ValueError(f'Cycle in {relation}')
                if key in done:
                    return
                active.add(key)
                for child in adjacency.get(key, []):
                    visit(child)
                active.remove(key)
                done.add(key)
            for key in adjacency:
                visit(key)
        return self

    def effective_concepts(self, model_id):
        """Explain composition and concept-broader closure, never lineage inheritance."""
        result, pending, seen = {}, [(model_id, [])], set()
        while pending:
            key, path = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            for edge in self.edges:
                if edge.source != key:
                    continue
                if edge.relation in ('uses', 'subconcept_of'):
                    trail = [*path, edge.claim_id]
                    result.setdefault(edge.target, trail)
                    pending.append((edge.target, trail))
                elif edge.relation == 'has_component':
                    pending.append((edge.target, [*path, edge.claim_id]))
        return result

    def compare(self, left, right):
        nodes = {node.id: node for node in self.nodes}
        if any(key not in nodes or nodes[key].kind != 'model' for key in (left, right)):
            raise ValueError('Choose two model nodes')
        def concepts(key):
            return set(self.effective_concepts(key))
        a, b = concepts(left), concepts(right)
        return {'left': nodes[left].model_dump(mode='json'), 'right': nodes[right].model_dump(mode='json'),
                'common': [nodes[key].model_dump(mode='json') for key in sorted(a & b)],
                'left_only': [nodes[key].model_dump(mode='json') for key in sorted(a - b)],
                'right_only': [nodes[key].model_dump(mode='json') for key in sorted(b - a)],
                'comparison_type': 'ontology_concepts',
                'facets': {relation: {'left': sorted({e.target for e in self.edges if e.source == left and e.relation == relation}),
                                      'right': sorted({e.target for e in self.edges if e.source == right and e.relation == relation})}
                           for relation in ('has_component', 'trained_with', 'applied_to')},
                'note': '구성 개념 비교이며 성능 비교나 직접적인 역사적 계승을 뜻하지 않습니다.'}

    def jsonld(self):
        import json
        from .ontology import data_graph, BASE
        context = {'@vocab': str(BASE), **{key: {'@id': str(BASE)+key+'/', '@prefix': True}
            for key in (*KINDS, 'claim')}}
        return json.loads(data_graph(self).serialize(format='json-ld', context=context, auto_compact=True))

    def query(self, *, concept=None, objective=None, task=None, family=None):
        selected = []
        known = {node.id: node for node in self.nodes}
        for key, kind in ((concept, 'concept'), (objective, 'objective'), (task, 'task')):
            if key is not None and (key not in known or known[key].kind != kind):
                raise ValueError(f'Unknown {kind}: {key}')
        for node in self.nodes:
            if node.kind != 'model' or (family and node.family != family):
                continue
            concepts = self.effective_concepts(node.id)
            if concept and concept not in concepts:
                continue
            if objective and not any(e.source == node.id and e.relation == 'trained_with' and e.target == objective for e in self.edges):
                continue
            if task and not any(e.source == node.id and e.relation == 'applied_to' and e.target == task for e in self.edges):
                continue
            proof = concepts.get(concept, [])
            by_claim = {edge.claim_id: edge for edge in self.edges}
            selected.append({'node': node.model_dump(mode='json'), 'proof': proof,
                'proof_edges': [by_claim[key].model_dump(mode='json') for key in proof]})
        return selected

    def neighborhood(self, node_id, *, hops=1, relations=None, limit=100):
        if not 1 <= hops <= 3 or not 1 <= limit <= 200:
            raise ValueError('hops must be 1..3 and limit 1..200')
        known = {node.id: node for node in self.nodes}
        if node_id not in known:
            raise ValueError('Unknown node')
        selected_relations = set(relations or RELATIONS)
        if selected_relations - RELATIONS.keys():
            raise ValueError('Unknown relationship')
        selected, frontier, truncated = {node_id}, {node_id}, False
        for _ in range(hops):
            next_frontier = set()
            for edge in self.edges:
                if edge.relation not in selected_relations:
                    continue
                if edge.source in frontier or edge.target in frontier:
                    for key in (edge.source, edge.target):
                        if key not in selected:
                            if len(selected) >= limit:
                                truncated = True
                            else:
                                selected.add(key)
                                next_frontier.add(key)
            frontier = next_frontier
        return {'nodes': [known[key].model_dump(mode='json') for key in sorted(selected)],
                'edges': [e.model_dump(mode='json') for e in self.edges if e.relation in selected_relations and e.source in selected and e.target in selected],
                'truncated': truncated, 'hops': hops}
