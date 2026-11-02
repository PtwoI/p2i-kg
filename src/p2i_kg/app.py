"""Local knowledge graph API and educational explorer."""
import asyncio
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .store import GraphStore
from .runtime import RuntimeSessions


def create_app(database=None):
    store = GraphStore(database or Path.home()/'.p2i'/'architecture-kg.sqlite3')
    sessions = RuntimeSessions()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(sessions.close_all)

    app = FastAPI(title='P2I Architecture Atlas', lifespan=lifespan)
    app.state.store, app.state.sessions = store, sessions

    def model_node(model_id):
        graph = store.load()
        node = next((node for node in graph.nodes if node.id == model_id and node.kind == 'model'), None)
        if node is None:
            raise HTTPException(404, '모델을 찾을 수 없습니다.')
        return node

    def local_request(request):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            raise HTTPException(403, '다른 사이트에서 P2I 세션을 시작할 수 없습니다.')

    @app.get('/api/kg')
    def graph():
        revision, graph = store.load_optional()
        return {'revision': revision, **graph.model_dump(mode='json')}

    @app.get('/api/kg/search')
    def search(q: str = Query('', max_length=200), kind: str | None = None):
        nodes = store.load().nodes
        if kind:
            nodes = [node for node in nodes if node.kind == kind]
        q = q.casefold().strip()
        return {'nodes': [node.model_dump(mode='json') for node in nodes if q in (node.label+' '+node.summary).casefold()]}

    @app.get('/api/kg/compare')
    def compare(left: str, right: str):
        try:
            return store.load().compare(left, right)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get('/api/kg/export')
    def export(format: str = 'json'):
        graph = store.load()
        if format not in ('json', 'jsonld'):
            raise HTTPException(422, 'format must be json or jsonld')
        return JSONResponse(graph.jsonld() if format == 'jsonld' else graph.model_dump(mode='json'),
            headers={'Content-Disposition': f'attachment; filename="p2i-architecture.{format}"'})

    @app.get('/api/kg/nodes/{node_id}')
    def node_detail(node_id: str):
        graph = store.load()
        node = next((node for node in graph.nodes if node.id == node_id), None)
        if node is None:
            raise HTTPException(404, '노드를 찾을 수 없습니다.')
        return {'node': node.model_dump(mode='json'),
                'edges': [edge.model_dump(mode='json') for edge in graph.edges if node_id in (edge.source, edge.target)]}

    @app.post('/api/kg/runtime/{model_id}')
    async def launch(model_id: str, request: Request):
        local_request(request)
        if 'application/json' not in request.headers.get('content-type',''):
            raise HTTPException(415, 'Use application/json')
        node = model_node(model_id)
        if not node.runtime_id:
            raise HTTPException(422, '실행 예제가 없는 모델입니다.')
        try:
            return await asyncio.to_thread(sessions.launch, node.runtime_id)
        except ImportError:
            raise HTTPException(503, 'P2I 실행 패키지를 설치한 뒤 다시 시도하세요. ResNet은 vision 의존성도 필요합니다.')
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    @app.delete('/api/kg/runtime/{model_id}')
    async def close(model_id: str, request: Request):
        local_request(request)
        node = model_node(model_id)
        return {'closed': await asyncio.to_thread(sessions.close, node.runtime_id)}

    @app.get('/api/kg/models/{model_id}/checkpoint')
    def checkpoint(model_id: str):
        node = model_node(model_id)
        try:
            import p2i
            from p2i.checkpoint import checkpoint_bytes
            from .runtime import make_model
            model, inputs = make_model(node.runtime_id)
            data = checkpoint_bytes(p2i.Harness(model, example_inputs=inputs))
            return Response(data, media_type='application/octet-stream',
                headers={'Content-Disposition': f'attachment; filename="{node.runtime_id}.p2i"', 'Cache-Control':'no-store'})
        except ImportError:
            raise HTTPException(503, '체크포인트를 만들려면 P2I 실행 패키지가 필요합니다.')

    app.mount('/', StaticFiles(directory=files('p2i_kg').joinpath('static'), html=True), name='explorer')
    return app
