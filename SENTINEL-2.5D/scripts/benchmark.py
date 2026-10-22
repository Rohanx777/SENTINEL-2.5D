"""Measure actual CUDA preprocessing, sparse inference, aggregation and transport.
No projected speedups; no downloads. Results describe local replay, not vehicle safety.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend/python_legacy'))
from runtime import read_scan,infer_cuda,encode_payload
from cylinder3d_net import build_model,CylinderVoxelizer
from cuda_runtime import CudaKernels
from foveated_grid_engine import FoveatedGridEngine
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scans',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--intensity-scale',type=float,default=1)
    p.add_argument('--frames',type=int,default=10)
    p.add_argument('--precision',choices=['fp32','tf32','fp16'],default='fp32')
    p.add_argument('--output',type=Path,default=ROOT/'benchmark-results.json')
    args=p.parse_args()
    if args.frames<1 or args.intensity_scale<=0: p.error('Positive frames and intensity scale required')
    scans=sorted(args.scans.glob('*.bin'))[:args.frames]
    if not scans: p.error('No scans')
    model=build_model(args.weights,torch.device('cuda'))
    if args.precision=='tf32':
        import spconv.constants
        spconv.constants.SPCONV_ALLOW_TF32=True
        torch.backends.cuda.matmul.allow_tf32=True
    if args.precision=='fp16':
        from inference_optimisation import optimise_model
        optimise_model(model,half=True)
    kernels=CudaKernels();grid=FoveatedGridEngine();rows=[]
    try:
        for idx,path in enumerate([scans[0]]*3+scans):
            torch.cuda.synchronize();start=time.perf_counter()
            points=read_scan(path);points[:,3]/=args.intensity_scale
            if points[:,3].min()<0 or points[:,3].max()>1.001: raise ValueError('Reflectance outside 0..1')
            uploaded=torch.from_numpy(points).cuda()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();fea,indices=kernels.voxelize(uploaded);b.record();b.synchronize()
            voxel_ms=a.elapsed_time(b)
            t=time.perf_counter();labels,semantic=infer_cuda(model,kernels,uploaded);torch.cuda.synchronize()
            inference_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();aggregates=kernels.aggregate(uploaded,labels)
            cpu=tuple(x.cpu().numpy() for x in aggregates)
            aggregation_ms=(time.perf_counter()-t)*1000
            t=time.perf_counter();cells,_=grid.project_aggregates(cpu,len(points),compact=True)
            message=encode_payload({'header':{'mode':'ml'},'telemetry':{},'grid_data':cells})
            packing_ms=(time.perf_counter()-t)*1000
            end=time.perf_counter()
            if idx>=3:
                semantic_cpu=semantic.cpu().numpy()
                rows.append(dict(scan=path.name,points=len(points),cells=len(cells),voxel_cuda_ms=voxel_ms,
                    inference_including_preprocess_ms=inference_ms,aggregation_and_readback_ms=aggregation_ms,
                    cell_assembly_and_json_ms=packing_ms,total_ms=(end-start)*1000,payload_bytes=len(message.encode()),
                    semantic_classes=np.unique(semantic_cpu).tolist()))
        report=dict(hardware=torch.cuda.get_device_name(0),python=platform.python_version(),torch=torch.__version__,precision=args.precision,
            checkpoint_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),intensity_scale=args.intensity_scale,
            warmup_frames=3,frame_count=len(rows),scope='Scan read, H2D, CUDA preprocessing, sparse neural inference, CUDA aggregation, D2H, cell assembly, JSON; excludes network and browser',
            limitations=['No ground-truth accuracy evaluation','Legacy checkpoint kernel migration; see vendor NOTICE','Python orchestration, CUDA C++ preprocessing/aggregation, spconv CUDA inference; not standalone C++ vehicle deployment'],
            median_total_ms=float(np.median([r['total_ms'] for r in rows])),p95_total_ms=float(np.percentile([r['total_ms'] for r in rows],95)),frames=rows)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
    finally: kernels.close()

if __name__=='__main__': main()
