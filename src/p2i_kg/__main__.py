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
    validation = sub.add_parser('validate')
    validation.add_argument('--rdf', action='store_true')
    export = sub.add_parser('export')
    export.add_argument('file')
    export.add_argument('--format', choices=['json','jsonld','turtle','ontology','shapes'], default='json')
    viewer = sub.add_parser('viewer')
    viewer.add_argument('directory')
    viewer.add_argument('--runtime', action='store_true')
    pages = sub.add_parser('pages')
    pages.add_argument('directory')
    pages.add_argument('--source-revision', required=True)
    verify = sub.add_parser('verify-pages')
    verify.add_argument('directory')
    plan = sub.add_parser('publication-plan')
    plan.add_argument('build')
    plan.add_argument('homepage')
    plan.add_argument('--expected-source', required=True)
    plan.add_argument('--expected-home', required=True)
    imported = sub.add_parser('import')
    imported.add_argument('file')
    extended = sub.add_parser('extend')
    extended.add_argument('file')
    extended.add_argument('--expected-revision')
    sub.add_parser('upgrade')
    queried = sub.add_parser('query')
    for field in ('concept', 'objective', 'task', 'family'):
        queried.add_argument('--' + field)
    opened = sub.add_parser('open')
    opened.add_argument('model_id')
    opened.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.command == 'publication-plan':
        from .publication_plan import publication_plan
        print(json.dumps(publication_plan(args.build, args.homepage,
            expected_source=args.expected_source, expected_home=args.expected_home), indent=2))
        return
    if args.command in ('pages', 'verify-pages'):
        from .publication import build_pages, verify_pages
        result = build_pages(args.directory, source_revision=args.source_revision) if args.command == 'pages' else verify_pages(args.directory)
        print(json.dumps(result))
        return
    store = GraphStore(args.database)
    if args.command == 'serve':
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(args.database), host='127.0.0.1', port=args.port)
    elif args.command == 'validate':
        graph = store.load()
        print(json.dumps({'valid':True,'nodes':len(graph.nodes),'edges':len(graph.edges)}))
        if args.rdf:
            from .ontology import validate_rdf
            report = validate_rdf(graph)
            print(json.dumps(report))
            if not report['conforms']:
                raise SystemExit(1)
    elif args.command == 'export':
        graph = store.load()
        if args.format in ('turtle', 'ontology', 'shapes'):
            from .ontology import data_graph, vocabulary, shapes
            rdf = vocabulary() if args.format == 'ontology' else shapes() if args.format == 'shapes' else data_graph(graph) + vocabulary()
            Path(args.file).write_text(rdf.serialize(format='turtle'), encoding='utf-8')
        else:
            value = graph.jsonld() if args.format == 'jsonld' else graph.model_dump(mode='json')
            Path(args.file).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    elif args.command == 'viewer':
        from .viewer import export_viewer
        print(json.dumps(export_viewer(store, args.directory, runtime=args.runtime)))
    elif args.command == 'import':
        value = KnowledgeGraph.model_validate_json(Path(args.file).read_text(encoding='utf-8'))
        print(json.dumps({'revision':store.replace(value)}))
    elif args.command == 'extend':
        print(json.dumps(store.extend(json.loads(Path(args.file).read_text(encoding='utf-8')), expected_revision=args.expected_revision)))
    elif args.command == 'upgrade':
        from .store import seed_graph
        print(json.dumps(store.extend(seed_graph().model_dump(mode='json'), keep_existing=True)))
    elif args.command == 'query':
        print(json.dumps(store.load().query(**{field:getattr(args,field) for field in ('concept','objective','task','family')}), ensure_ascii=False, indent=2))
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
