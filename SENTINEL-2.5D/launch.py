"""Start owned SENTINEL services; never kills existing listeners or installs packages."""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
ROOT=Path(__file__).resolve().parent

def available(port):
    with socket.socket() as sock:
        try: sock.bind(('127.0.0.1',port))
        except OSError: raise RuntimeError(f'Port {port} occupied. Stop its owner or choose another port.')

def wait_ready(process,port,timeout=90,http=False):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError(f'Service exited with code {process.returncode}')
        try:
            if http:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}',timeout=2) as response:
                    if response.status==200: return
            else:
                with socket.create_connection(('127.0.0.1',port),timeout=1): return
        except (OSError,urllib.error.URLError): time.sleep(0.25)
    raise TimeoutError(f'Service on {port} did not become ready')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','replay','ml'],default='demo')
    p.add_argument('--scans',type=Path,default=ROOT/'data/sequences/00/velodyne')
    p.add_argument('--weights',type=Path,default=ROOT/'weights/model_load.pth')
    p.add_argument('--port',type=int,default=8005)
    p.add_argument('--frontend-port',type=int,default=3000)
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--intensity-scale',type=float,default=1.0)
    p.add_argument('--precision',choices=['fp32','tf32','fp16'],default='fp32')
    args=p.parse_args()
    if args.port==args.frontend_port: p.error('Service ports must differ')
    for port in (args.port,args.frontend_port):
        if not 1<=port<=65535: p.error('Port outside 1..65535')
        available(port)
    frontend=ROOT/'frontend'
    if not (frontend/'node_modules/next').exists(): p.error('Run npm ci in frontend first; see README for Python dependencies.')
    npm=shutil.which('npm.cmd' if os.name=='nt' else 'npm')
    if not npm: p.error('Node.js/npm not found')
    cmd=[sys.executable,'-u',str(ROOT/'scripts/serve.py'),'--mode',args.mode,'--scans',str(args.scans.resolve()),'--weights',str(args.weights.resolve()),'--port',str(args.port),'--intensity-scale',str(args.intensity_scale)]
    processes=[]
    cmd+=['--precision',args.precision]
    try:
        backend=subprocess.Popen(cmd,cwd=ROOT); processes.append(backend)
        wait_ready(backend,args.port)
        env=dict(os.environ,NEXT_PUBLIC_SENTINEL_WS_URL=f'ws://127.0.0.1:{args.port}/ws/stream_map')
        ui=subprocess.Popen([npm,'run','dev','--','--hostname','127.0.0.1','--port',str(args.frontend_port)],cwd=frontend,env=env); processes.append(ui)
        wait_ready(ui,args.frontend_port,http=True)
        url=f'http://127.0.0.1:{args.frontend_port}'
        print(f'SENTINEL ready: {url} | mode={args.mode}',flush=True)
        if not args.no_browser: webbrowser.open(url)
        while all(proc.poll() is None for proc in processes): time.sleep(0.5)
        raise RuntimeError('A service stopped; shutting down owned services')
    except KeyboardInterrupt: pass
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                if os.name=='nt': subprocess.run(['taskkill','/F','/T','/PID',str(proc.pid)],capture_output=True)
                else: proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill()

if __name__=='__main__':
    try: main()
    except (RuntimeError,TimeoutError) as exc:
        print(f'SENTINEL: {exc}',file=sys.stderr); sys.exit(1)
