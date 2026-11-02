"""Small typed ontology; relationship evidence and interpretation stay explicit."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Node(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-z0-9:_-]+$')
    kind: Literal['model', 'concept', 'paper']
    label: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    reference_year: int | None = Field(default=None, ge=1900, le=2100)
    family: str | None = None
    url: HttpUrl | None = None
    authors: list[str] = Field(default_factory=list)
    runtime_id: str | None = None
    runtime_note: str | None = None


class Edge(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str
    relation: Literal['uses', 'documented_in', 'extends', 'alternative_to']
    target: str
    basis: Literal['documented', 'structural', 'pedagogical']
    explanation: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)


class KnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['0.1'] = '0.1'
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
            if nodes[edge.source].kind != 'model':
                raise ValueError('Relationship source must be a model')
            target_kind = {'uses':'concept', 'documented_in':'paper', 'extends':'model', 'alternative_to':'model'}[edge.relation]
            if nodes[edge.target].kind != target_kind:
                raise ValueError('Relationship domain/range mismatch')
            if any(key not in nodes or nodes[key].kind != 'paper' for key in edge.evidence):
                raise ValueError('Relationship evidence must reference paper nodes')
        for node in self.nodes:
            if node.kind == 'paper' and (node.url is None or not node.authors or node.reference_year is None):
                raise ValueError('Paper nodes require URL, authors and reference year')
            if node.kind != 'model' and node.runtime_id:
                raise ValueError('Only models can have runtime implementations')
        return self

    def compare(self, left, right):
        nodes = {node.id: node for node in self.nodes}
        if any(key not in nodes or nodes[key].kind != 'model' for key in (left, right)):
            raise ValueError('Choose two model nodes')
        def concepts(key):
            return {edge.target for edge in self.edges if edge.source == key and edge.relation == 'uses'}
        a, b = concepts(left), concepts(right)
        return {'left': nodes[left].model_dump(mode='json'), 'right': nodes[right].model_dump(mode='json'),
                'common': [nodes[key].model_dump(mode='json') for key in sorted(a & b)],
                'left_only': [nodes[key].model_dump(mode='json') for key in sorted(a - b)],
                'right_only': [nodes[key].model_dump(mode='json') for key in sorted(b - a)],
                'comparison_type': 'curated_concepts',
                'note': '구성 개념 비교이며 성능 비교나 직접적인 역사적 계승을 뜻하지 않습니다.'}

    def jsonld(self):
        base = 'https://github.com/PtwoI/p2i-kg#'
        graph = []
        for node in self.nodes:
            item = {'@id': node.id, '@type': node.kind.capitalize(), 'label': node.label, 'summary': node.summary}
            for edge in self.edges:
                if edge.source == node.id:
                    item.setdefault(edge.relation, []).append({'@id': edge.target})
            if node.url:
                item['url'] = str(node.url)
            graph.append(item)
        # Reified claims retain evidence and interpretation when exported.
        for i, edge in enumerate(self.edges):
            graph.append({'@id': f'claim:{i}', '@type': 'Claim', 'source': {'@id': edge.source},
                          'relation': {'@id': base + edge.relation}, 'target': {'@id': edge.target},
                          'basis': edge.basis, 'explanation': edge.explanation,
                          'evidence': [{'@id': key} for key in edge.evidence]})
        return {'@context': {'@vocab': base, **{key: {'@id': base+key+'/', '@prefix': True} for key in ('model','concept','paper','claim')}}, '@graph': graph}
