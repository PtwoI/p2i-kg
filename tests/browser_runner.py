import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode='w+') as log:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    server = subprocess.Popen([sys.executable,'-m','p2i_kg','--database',str(Path(directory)/'atlas.sqlite3'),'serve','--port',str(port)],stdout=log,stderr=log)
    try:
        for _ in range(300):
            if server.poll() is not None:raise RuntimeError('KG server exited before startup')
            try:
                with urlopen(url+'/api/kg',timeout=1) as response:
                    if response.status == 200:break
            except OSError:time.sleep(.2)
        else:raise RuntimeError('KG server did not become ready')
        subprocess.run(['node','tests/browser.mjs'],cwd=root,env={**os.environ,'P2I_KG_URL':url},check=True,timeout=180)
    except Exception:
        log.seek(0);print(log.read(),file=sys.stderr);raise
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
