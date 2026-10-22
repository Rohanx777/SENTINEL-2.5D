import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend/python_legacy'))
from runtime import read_scan,encode_payload
from foveated_grid_engine import FoveatedGridEngine

def load_server():
    spec=importlib.util.spec_from_file_location('sentinel_service',ROOT/'scripts/serve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_scan_validation(tmp_path):
    path=tmp_path/'frame.bin'
    path.write_bytes(b'bad')
    with pytest.raises(ValueError): read_scan(path)
    np.array([[1,2,3,np.nan]],dtype='<f4').tofile(path)
    with pytest.raises(ValueError): read_scan(path)
    np.array([[1,2,3,0.5]],dtype='<f4').tofile(path)
    assert read_scan(path).shape==(1,4)

def test_exact_transport_size():
    message=encode_payload({'telemetry':{},'description':'LiDAR — synthetic'})
    assert json.loads(message)['telemetry']['payload_bytes']==len(message.encode())

def test_angular_boundary_and_polygon_units():
    points=np.array([[-2.,0.,-1.6,0.2],[-2.,0.,-1.6,0.2]],dtype=np.float32)
    cells,_=FoveatedGridEngine().project_point_cloud(points,np.zeros(2,dtype=np.uint8))
    assert len(cells)==1
    assert cells[0]['cx']<0
    assert all(x<0 for x,y in cells[0]['polygon'])
    assert cells[0]['point_density']==2

def test_model_rejects_missing_checkpoint(tmp_path):
    torch=pytest.importorskip('torch')
    from cylinder3d_net import build_model
    with pytest.raises(FileNotFoundError): build_model(tmp_path/'missing.pt',torch.device('cpu'))

def test_demo_truth_and_single_representation():
    service=load_server().StreamService()
    payload=service.make_frame()
    assert payload['header']['mode']=='demo'
    assert payload['header']['model_loaded'] is False
    assert payload['header']['active_engine']=='SYNTHETIC DEMO'
    assert payload['telemetry']['raw_points_count']==15000
    assert 'polar_grid' not in payload and 'foveated_grid' not in payload
    assert payload['telemetry']['latency_ms']>0
    assert all(z['accuracy']=='Not evaluated' for z in payload['stratified_metrics']['zones'])

def test_stream_and_playback():
    import websockets
    async def scenario():
        service=load_server().StreamService(fps=10)
        async with websockets.serve(service.handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            async with websockets.connect(f'ws://127.0.0.1:{port}/ws/stream_map',max_size=5_000_000) as ws:
                task=asyncio.create_task(service.loop())
                try:
                    frame=json.loads(await asyncio.wait_for(ws.recv(),10))
                    assert frame['header']['mode']=='demo'
                    await ws.send(json.dumps({'action':'set_speed','speed':4}))
                    for _ in range(5):
                        frame=json.loads(await asyncio.wait_for(ws.recv(),10))
                        if frame['telemetry']['playback_speed']==4: break
                    assert frame['telemetry']['playback_speed']==4
                    await ws.send('{"action":"set_speed","speed":"NaN"}')
                    for _ in range(5):
                        result=json.loads(await asyncio.wait_for(ws.recv(),10))
                        if 'error' in result: break
                    assert 'error' in result
                finally:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(scenario())

def test_missing_replay_does_not_become_demo(tmp_path):
    with pytest.raises(FileNotFoundError): load_server().StreamService('replay',tmp_path)

def test_slow_client_is_removed_without_stopping_other_clients():
    class Client:
        def __init__(self, slow=False):
            self.slow=slow; self.closed=False; self.messages=[]
        async def send(self,message):
            if self.slow: raise asyncio.TimeoutError()
            self.messages.append(message)
        async def close(self,code,reason):
            self.closed=True
    async def scenario():
        service=load_server().StreamService()
        slow=Client(slow=True); healthy=Client()
        service.clients.update([slow,healthy])
        await asyncio.gather(service.send(slow,'frame'),service.send(healthy,'frame'))
        assert slow.closed and slow not in service.clients
        assert healthy in service.clients and healthy.messages==['frame']
    asyncio.run(scenario())
