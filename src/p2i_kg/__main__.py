import argparse
import json
from pathlib import Path
from .store import GraphStore
from .schema import KnowledgeGraph


def main():
    parser = argparse.ArgumentParser(description='P2I Architecture Atlas')
    parser.add_argument('--database', default=str(Path.home()/'.p2i'/'architecture-kg.sqlite3'))
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--port', type=int, default=8010)
    sub.add_parser('validate')
    export = sub.add_parser('export')
    export.add_argument('file')
    export.add_argument('--format', choices=['json','jsonld'], default='json')
    imported = sub.add_parser('import')
    imported.add_argument('file')
    opened = sub.add_parser('open')
    opened.add_argument('model_id')
    opened.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    store = GraphStore(args.database)
    if args.command == 'serve':
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(args.database), host='127.0.0.1', port=args.port)
    elif args.command == 'validate':
        graph = store.load()
        print(json.dumps({'valid':True,'nodes':len(graph.nodes),'edges':len(graph.edges)}))
    elif args.command == 'export':
        graph = store.load()
        value = graph.jsonld() if args.format == 'jsonld' else graph.model_dump(mode='json')
        Path(args.file).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    elif args.command == 'import':
        value = KnowledgeGraph.model_validate_json(Path(args.file).read_text(encoding='utf-8'))
        print(json.dumps({'revision':store.replace(value)}))
    else:
        from .runtime import make_model
        import p2i
        key = args.model_id if args.model_id.startswith('model:') else 'model:'+args.model_id
        node = next((node for node in store.load().nodes if node.id == key and node.kind == 'model'), None)
        if node is None:
            parser.error('Unknown model ID')
        model, inputs = make_model(node.runtime_id)
        p2i.serve(p2i.Harness(model, example_inputs=inputs), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
