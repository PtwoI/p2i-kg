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
