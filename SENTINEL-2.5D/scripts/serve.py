"""Explicit demo/replay/ML WebSocket service with measured metrics and bounded sends."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import websockets
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend/python_legacy'))
from coordinate_transformer import OdometryPose
from foveated_grid_engine import FoveatedGridEngine
from comparative_analyzer import ComparativeMetrics,sample_point_cloud_3d
from runtime import read_scan,geometric_labels,infer_points,infer_cuda,encode_payload

class StreamService:
    def __init__(self,mode='demo',scans=None,weights=None,fps=30,preview=True,intensity_scale=1.0,precision='fp32'):
        self.mode=mode; self.preview=preview; self.target_fps=fps; self.speed=1.0
        self.clients=set(); self.frame_id=0; self.previous_time=None
        self.intensity_scale=intensity_scale
        self.grid=FoveatedGridEngine(history_len=1) # No uncalibrated cross-frame world smoothing.
        self.scans=[]; self.model=None; self.device=None; self.weights=weights
        if mode!='demo':
            folder=Path(scans) if scans else ROOT/'data/sequences/00/velodyne'
            self.scans=sorted(folder.glob('*.bin'))
            if not self.scans: raise FileNotFoundError(f'No .bin scans in {folder}')
            read_scan(self.scans[0])
        if mode=='ml':
            import torch
            from cylinder3d_net import build_model,CylinderVoxelizer
            self.device=torch.device('cuda')
            self.model=build_model(weights,self.device)
            if precision=='tf32':
                import spconv.constants
                spconv.constants.SPCONV_ALLOW_TF32=True
                torch.backends.cuda.matmul.allow_tf32=True
            if precision=='fp16':
                from inference_optimisation import optimise_model
                optimise_model(self.model,half=True)
            self.voxelizer=CylinderVoxelizer()
            from cuda_runtime import CudaKernels
            self.kernels=CudaKernels()

    def points(self):
        if self.mode!='demo':
            points=read_scan(self.scans[self.frame_id%len(self.scans)])
            points[:,3]/=self.intensity_scale
            if self.mode=='ml' and (points[:,3].min()<0 or points[:,3].max()>1.001):
                raise ValueError('ML reflectance must be 0..1. For byte intensities use --intensity-scale 255.')
            return points
        # Deterministic synthetic ground/building/vehicle-like points; labels are fixtures.
        rng=np.random.default_rng(self.frame_id)
        ground=np.column_stack([rng.uniform(-40,40,12000),rng.uniform(-20,20,12000),rng.normal(-1.6,0.02,12000),rng.uniform(0,1,12000)])
        wall=np.column_stack([rng.uniform(8,12,2000),rng.uniform(6,10,2000),rng.uniform(-1.6,4,2000),rng.uniform(0,1,2000)])
        vehicle=np.column_stack([rng.uniform(12,16,1000)-self.frame_id*0.05,rng.uniform(-2,0,1000),rng.uniform(-1.5,0.2,1000),rng.uniform(0,1,1000)])
        return np.vstack([ground,wall,vehicle]).astype(np.float32)

    def make_frame(self):
        started=time.perf_counter()
        points=self.points()
        if self.mode=='ml':
            import torch
            gpu_points=torch.from_numpy(points).to(self.device)
            gpu_labels,_=infer_cuda(self.model,self.kernels,gpu_points)
            aggregates=self.kernels.aggregate(gpu_points,gpu_labels)
            cpu_aggregates=tuple(t.cpu().numpy() for t in aggregates)
            labels=gpu_labels.cpu().numpy()
        elif self.mode=='demo': labels=np.concatenate([np.zeros(12000),np.ones(2000),np.full(1000,2)]).astype(np.uint8)
        else: labels=geometric_labels(points)
        inference_ms=(time.perf_counter()-started)*1000
        if self.mode=='ml': cells,stratified=self.grid.project_aggregates(cpu_aggregates,len(points),OdometryPose(),compact=True)
        else: cells,stratified=self.grid.project_point_cloud(points,labels,OdometryPose(),compact=True)
        comparative=ComparativeMetrics.compute(points,cells)
        processing_ms=(time.perf_counter()-started)*1000
        now=time.perf_counter()
        observed_fps=0.0 if self.previous_time is None else 1/max(now-self.previous_time,1e-9)
        self.previous_time=now
        grid_bytes=len(json.dumps(cells,separators=(',',':')).encode())
        raw_bytes=points.nbytes
        reduction=100*(1-grid_bytes/max(raw_bytes,1))
        payload={
            'header':{'frame_id':self.frame_id,'timestamp':time.time(),'active_engine':{'demo':'SYNTHETIC DEMO','replay':'GEOMETRY BASELINE','ml':'CYLINDER3D CUDA'}[self.mode],
                'mode':self.mode,'data_source':'synthetic' if self.mode=='demo' else 'KITTI replay','inference_method':{'demo':'fixture labels','replay':'height heuristic','ml':'pretrained Cylinder3D'}[self.mode],
                'model_loaded':self.model is not None,'coordinate_frame':'sensor','metrics_measured':True,'grid_encoding':'polar-v1'},
            'telemetry':{'fps':observed_fps,'target_fps':self.target_fps,'latency_ms':processing_ms,'inference_ms':inference_ms,
                'raw_points_count':len(points),'compressed_cells_count':len(cells),'memory_saved_percent':reduction,
                'raw_bytes':raw_bytes,'grid_bytes':grid_bytes,'playback_speed':self.speed},
            'grid_data':cells,'pointcloud_3d':sample_point_cloud_3d(points,labels,1500) if self.preview else [],
            'comparative_study':comparative,'stratified_metrics':stratified,'threats':[],
        }
        self.frame_id+=max(1,round(self.speed))
        return payload

    async def handler(self,ws):
        if ws.request.path!='/ws/stream_map':
            await ws.close(1008,'Unknown stream route'); return
        self.clients.add(ws)
        try:
            async for message in ws:
                try:
                    data=json.loads(message)
                    if data.get('action')=='set_speed':
                        speed=float(data['speed'])
                        if not math.isfinite(speed) or not 1<=speed<=8: raise ValueError('Speed must be 1..8')
                        self.speed=speed
                    else: raise ValueError('Unknown command')
                except (ValueError,TypeError,KeyError):
                    await ws.send(json.dumps({'error':'Invalid playback command'}))
        finally: self.clients.discard(ws)

    async def send(self,ws,message):
        try: await asyncio.wait_for(ws.send(message),timeout=0.5)
        except (asyncio.TimeoutError,websockets.ConnectionClosed):
            self.clients.discard(ws)
            await ws.close(1013,'Slow or disconnected client')

    async def loop(self):
        while True:
            if not self.clients:
                await asyncio.sleep(0.05); continue
            start=time.perf_counter()
            payload=await asyncio.to_thread(self.make_frame)
            message=encode_payload(payload)
            await asyncio.gather(*(self.send(ws,message) for ws in tuple(self.clients)))
            await asyncio.sleep(max(0.001,1/self.target_fps-(time.perf_counter()-start)))

async def run(args):
    service=StreamService(args.mode,args.scans,args.weights,args.fps,not args.grid_only,args.intensity_scale,args.precision)
    # Preflight model/data before announcing readiness, including a real forward pass.
    await asyncio.to_thread(service.make_frame)
    async with websockets.serve(service.handler,'127.0.0.1',args.port,max_size=4096,max_queue=4,write_limit=65536):
        print(f'Ready: mode={args.mode} ws://127.0.0.1:{args.port}/ws/stream_map',flush=True)
        await service.loop()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['demo','replay','ml'],default='demo')
    p.add_argument('--scans',type=Path,default=ROOT/'data/sequences/00/velodyne')
    p.add_argument('--weights',type=Path,default=ROOT/'weights/model_load.pth')
    p.add_argument('--port',type=int,default=8005)
    p.add_argument('--fps',type=float,default=30)
    p.add_argument('--precision',choices=['fp32','tf32','fp16'],default='fp32')
    p.add_argument('--grid-only',action='store_true')
    p.add_argument('--intensity-scale',type=float,default=1.0)
    args=p.parse_args()
    if not math.isfinite(args.fps) or not 1<=args.fps<=60: p.error('FPS must be 1..60')
    if not 1<=args.port<=65535: p.error('Port outside 1..65535')
    if not math.isfinite(args.intensity_scale) or args.intensity_scale<=0: p.error('Intensity scale must be positive')
    try: asyncio.run(run(args))
    except KeyboardInterrupt: pass
    except Exception as exc:
        print(f'SENTINEL failed: {exc}',file=sys.stderr); sys.exit(1)

if __name__=='__main__': main()
