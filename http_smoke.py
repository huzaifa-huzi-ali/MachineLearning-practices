"""Bounded local HTTP check with guaranteed cleanup of its own server process."""
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
log = ROOT / 'reports/http_smoke.log'
with log.open('w') as stream:
    process = subprocess.Popen([
        'python', '-m', 'streamlit', 'run', 'app.py', '--server.headless', 'true',
        '--server.address', '127.0.0.1', '--server.port', '8502'],
        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    try:
        ready = False
        for _ in range(60):
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen('http://127.0.0.1:8502/_stcore/health', timeout=1) as response:
                    if response.status == 200:
                        print('Health:', response.status, response.read().decode())
                        ready = True
                        break
            except (OSError, TimeoutError):
                time.sleep(0.25)
        if not ready:
            raise RuntimeError('Local server readiness failed; inspect reports/http_smoke.log.')
        with urllib.request.urlopen('http://127.0.0.1:8502/?task=heart', timeout=5) as response:
            body = response.read()
            assert response.status == 200 and len(body) > 0
            print('App HTML:', response.status, 'bytes:', len(body))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
print('Owned server stopped.')
