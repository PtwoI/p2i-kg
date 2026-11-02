"""Transactional SQLite catalog storage with content-addressed revisions."""
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
import json
import sqlite3
from .schema import KnowledgeGraph


def seed_graph():
    return KnowledgeGraph.model_validate_json(files('p2i_kg').joinpath('catalog.json').read_text(encoding='utf-8'))


class GraphStore:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS catalog (id INTEGER PRIMARY KEY CHECK(id=1), revision TEXT NOT NULL, body TEXT NOT NULL)')
        if self.load_optional() is None:
            self.replace(seed_graph())

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def load_optional(self):
        with self.connect() as connection:
            row = connection.execute('SELECT revision,body FROM catalog WHERE id=1').fetchone()
        return None if row is None else (row[0], KnowledgeGraph.model_validate_json(row[1]))

    def load(self):
        return self.load_optional()[1]

    def replace(self, graph):
        # Validate fully before opening a write transaction.
        graph = KnowledgeGraph.model_validate(graph.model_dump() if isinstance(graph, KnowledgeGraph) else graph)
        body = json.dumps(graph.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
        revision = sha256(body.encode()).hexdigest()
        with self.connect() as connection:
            connection.execute('INSERT INTO catalog(id,revision,body) VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,body=excluded.body', (revision,body))
        return revision


    def extend(self, patch, *, expected_revision=None, keep_existing=False):
        """Atomically add a validated ontology fragment without replacing curation."""
        if not isinstance(patch, dict) or set(patch) - {'schema_version', 'title', 'nodes', 'edges'}:
            raise ValueError('Patch must contain only schema_version, title, nodes and edges')
        if patch.get('schema_version', '0.2') not in ('0.1', '0.2'):
            raise ValueError('Unsupported patch schema version')
        from .schema import Node, Edge
        incoming_nodes = [Node.model_validate(value) for value in patch.get('nodes', [])]
        incoming_edges = [Edge.model_validate(value) for value in patch.get('edges', [])]
        if len({n.id for n in incoming_nodes}) != len(incoming_nodes):
            raise ValueError('Duplicate node ID in patch')
        signature = lambda edge: (edge.source, edge.relation, edge.target)
        if len({signature(e) for e in incoming_edges}) != len(incoming_edges):
            raise ValueError('Duplicate relationship in patch')
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            revision, body = connection.execute('SELECT revision,body FROM catalog WHERE id=1').fetchone()
            if expected_revision is not None and revision != expected_revision:
                raise ValueError('Stale catalog revision; reload before extending')
            current = KnowledgeGraph.model_validate_json(body)
            nodes, edges = {n.id: n for n in current.nodes}, {signature(e): e for e in current.edges}
            conflicts = []
            for key, old, new, label in [(n.id, nodes.get(n.id), n, 'node') for n in incoming_nodes] + [
                    (signature(e), edges.get(signature(e)), e, 'relationship') for e in incoming_edges]:
                if old is not None and old != new:
                    if keep_existing:
                        conflicts.append(str(key))
                        continue
                    raise ValueError(f'Conflicting {label}: {key}')
                (nodes if label == 'node' else edges)[key] = new
            merged = KnowledgeGraph(title=current.title, nodes=list(nodes.values()), edges=list(edges.values()))
            body = json.dumps(merged.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
            updated = sha256(body.encode()).hexdigest()
            connection.execute('UPDATE catalog SET revision=?,body=? WHERE id=1', (updated, body))
        return {'revision': updated, 'added_nodes': len(merged.nodes)-len(current.nodes),
                'added_edges': len(merged.edges)-len(current.edges), 'retained_conflicts': conflicts}
