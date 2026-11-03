"""Fixed educational model factories and bounded loopback P2I sessions."""
import socket
from threading import Lock, Thread
import time

from .extended_registry import EXTENDED_IDS

BASE_RUNTIME_IDS = ('perceptron', 'mlp', 'cnn', 'rnn', 'lstm', 'gru', 'resnet18',
    'transformer', 'seq2seq', 'bert', 'vit', 'gpt2', 'mixer', 'mae', 'detr',
    'nerf', 'perceiver', 'graphsage', 'deepsets', 'pointnet', 'simclr', 'byol',
    'ldm', 'dit', 'moco', 'dino', 'gin', 'mamba2')
RUNTIME_IDS = BASE_RUNTIME_IDS + EXTENDED_IDS


def make_model(runtime_id):
    if runtime_id not in RUNTIME_IDS:
        raise ValueError('Unknown runtime implementation')
    import torch
    from torch import nn
    from p2i.examples import make_demo
    # Fork RNG so catalog exploration does not alter an application's RNG state.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(47)
        if runtime_id == 'perceptron':
            from .models import ThresholdPerceptron
            return ThresholdPerceptron().eval(), (torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),)
        if runtime_id in ('mixer', 'mae'):
            from .models import TinyMixer, TinyMAE
            model = TinyMixer() if runtime_id == 'mixer' else TinyMAE()
            return model.eval(), (torch.randn(2, 3, 16, 16),)
        if runtime_id == 'detr':
            from .models import TinyDETR
            return TinyDETR().eval(), (torch.randn(2, 3, 16, 16),)
        if runtime_id == 'nerf':
            from .models import TinyNeRF
            # Two rays with six ordered samples and unit-length intervals.
            return TinyNeRF().eval(), (torch.randn(2, 6, 3), torch.randn(2, 6, 3), torch.ones(2, 6))
        if runtime_id == 'perceiver':
            from .models import TinyPerceiver
            return TinyPerceiver().eval(), (torch.randn(2, 12, 8),)
        if runtime_id == 'graphsage':
            from .models import TinyGraphSAGE
            adjacency = torch.tensor([[[0., 1., 1., 0., 0.],
                [1., 0., 1., 0., 0.], [1., 1., 0., 1., 0.],
                [0., 0., 1., 0., 1.], [0., 0., 0., 1., 0.]]]).expand(2, -1, -1)
            return TinyGraphSAGE().eval(), (torch.randn(2, 5, 8), adjacency)
        if runtime_id in ('deepsets', 'pointnet'):
            from .models import TinyDeepSets, TinyPointNet
            if runtime_id == 'deepsets':
                return TinyDeepSets().eval(), (torch.randn(2, 7, 6),)
            return TinyPointNet().eval(), (torch.randn(2, 16, 3),)
        if runtime_id in ('simclr', 'byol'):
            from .models import TinySimCLR, TinyBYOL
            model = TinySimCLR() if runtime_id == 'simclr' else TinyBYOL()
            # The two tensors represent independently augmented views of the
            # same batch; augmentation itself remains outside the fixed graph.
            return model.eval(), (torch.randn(4, 3, 16, 16),
                                  torch.randn(4, 3, 16, 16))
        if runtime_id in ('moco', 'dino'):
            from .models import TinyMoCo, TinyDINO
            model = TinyMoCo() if runtime_id == 'moco' else TinyDINO()
            return model.eval(), (torch.randn(4, 3, 16, 16),
                                  torch.randn(4, 3, 16, 16))
        if runtime_id == 'gin':
            from .models import TinyGIN
            adjacency = torch.tensor([[[0., 1., 1., 0., 0.],
                [1., 0., 1., 0., 0.], [1., 1., 0., 1., 0.],
                [0., 0., 1., 0., 1.], [0., 0., 0., 1., 0.]]]).expand(2, -1, -1)
            return TinyGIN().eval(), (torch.randn(2, 5, 8), adjacency)
        if runtime_id == 'mamba2':
            from .models import TinyMamba2
            return TinyMamba2().eval(), (torch.randn(2, 8, 8),)
        if runtime_id == 'ldm':
            from .models import TinyLatentDiffusion
            return TinyLatentDiffusion().eval(), (
                torch.randn(2, 3, 16, 16), torch.randn(2, 4, 8),
                torch.randn(2, 8, 8, 8), torch.tensor([2, 5]))
        if runtime_id == 'dit':
            from .models import TinyDiT
            return TinyDiT().eval(), (
                torch.randn(2, 4, 8, 8), torch.tensor([2, 5]),
                torch.tensor([1, 3]))
        if runtime_id == 'resnet18':
            from torchvision.models import resnet18
            return resnet18(weights=None, num_classes=4).eval(), (torch.randn(2, 3, 32, 32),)
        if runtime_id in ('bert', 'vit'):
            from p2i.verification import make_case
            model, inputs, _ = make_case(runtime_id)
            return model, inputs
        if runtime_id == 'gpt2':
            from transformers import GPT2Config, GPT2LMHeadModel
            config = GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1,
                n_head=2, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                use_cache=False, return_dict=False, attn_implementation='eager')
            return GPT2LMHeadModel(config).eval(), (torch.randint(0, 32, (2, 6)),)
        if runtime_id in EXTENDED_IDS:
            from .extended_models import make_extended
            return make_extended(runtime_id)
        return make_demo(runtime_id)


class RuntimeSessions:
    def __init__(self, maximum=3):
        self.maximum = maximum
        self._sessions = {}
        self._lock = Lock()

    def launch(self, runtime_id):
        with self._lock:
            old = self._sessions.get(runtime_id)
            if old and old['thread'].is_alive():
                return {'url': old['url'], 'runtime_id': runtime_id, 'reused': True}
            if old:
                old['socket'].close()
                self._sessions.pop(runtime_id)
            if len(self._sessions) >= self.maximum:
                raise ValueError('세 개의 P2I 세션이 열려 있습니다. 세션을 닫은 뒤 새 모델을 여세요.')
            import p2i
            from p2i_gui.app import create_app
            import uvicorn
            model, inputs = make_model(runtime_id)
            # Runtime tracing is sufficient to open the model; static export is
            # optional and avoids delaying the interactive launch unnecessarily.
            harness = p2i.Harness(model, example_inputs=inputs)
            harness.retrace(static=False)
            app = create_app(harness)
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
            thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
            url = f'http://127.0.0.1:{port}'
            try:
                thread.start()
                deadline = time.monotonic() + 10
                while thread.is_alive() and not server.started and time.monotonic() < deadline:
                    time.sleep(0.02)
                if not server.started:
                    raise RuntimeError('P2I server did not start')
            except BaseException:
                server.should_exit = True
                thread.join(timeout=2)
                sock.close()
                raise
            self._sessions[runtime_id] = {'server': server, 'thread': thread, 'socket': sock, 'url': url}
            return {'url': url, 'runtime_id': runtime_id, 'reused': False}

    def close(self, runtime_id):
        with self._lock:
            item = self._sessions.pop(runtime_id, None)
            if item:
                item['server'].should_exit = True
                item['thread'].join(timeout=5)
                item['socket'].close()
            return item is not None

    def close_all(self):
        for key in list(self._sessions):
            self.close(key)
