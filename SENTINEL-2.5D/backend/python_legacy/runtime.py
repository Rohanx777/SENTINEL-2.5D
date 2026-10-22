"""Validated scan input, measured transport, and trained point inference."""
import json
from pathlib import Path
import numpy as np

def read_scan(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0 or path.stat().st_size%16:
        raise ValueError(f'{path}: expected nonempty little-endian float32 KITTI x,y,z,intensity records')
    points=np.fromfile(path,dtype='<f4').reshape(-1,4)
    if not np.isfinite(points).all(): raise ValueError(f'{path}: nonfinite coordinates')
    return points

def geometric_labels(points):
    # Explicitly a geometry baseline, not semantic prediction or a motion detector.
    return np.where(points[:,2]>-1.1,1,0).astype(np.uint8)

def infer_points(model,voxelizer,points,device):
    import torch
    if len(points)==0: return np.empty(0,dtype=np.uint8),np.empty(0,dtype=np.uint8)
    with torch.inference_mode():
        features,indices=voxelizer(points)
        features,indices=features.to(device),indices.to(device)
        logits=model([features],[indices],1)
        if not torch.isfinite(logits).all(): raise RuntimeError('Nonfinite model output; refusing to publish labels')
        dense=logits[0].argmax(0)
        semantic=dense[indices[:,0].long(),indices[:,1].long(),indices[:,2].long()]
        remap=torch.tensor([0,2,2,2,2,2,2,2,2,0,0,0,0,1,1,0,1,0,1,1],dtype=torch.uint8,device=device)
        return remap[semantic].cpu().numpy(),semantic.cpu().numpy().astype(np.uint8)

def infer_cuda(model,kernels,points):
    import torch
    with torch.inference_mode():
        features,indices=kernels.voxelize(points)
        features=features.to(model.cylinder_3d_generator.PPmodel[1].weight.dtype)
        coords,voxel_features=model.cylinder_3d_generator([features],[indices])
        output=model.cylinder_3d_spconv_seg(voxel_features,coords,1,return_sparse=True)
        if not torch.isfinite(output.features).all(): raise RuntimeError('Nonfinite model output')
        # Avoid a 480 x 360 x 32 x 20 dense tensor: map occupied voxel rows only.
        dims=model.sparse_shape
        keys=(output.indices[:,1].long()*int(dims[1])+output.indices[:,2])*int(dims[2])+output.indices[:,3]
        order=torch.argsort(keys)
        point_keys=(indices[:,0].long()*int(dims[1])+indices[:,1])*int(dims[2])+indices[:,2]
        rows=torch.searchsorted(keys[order],point_keys)
        if torch.any(rows>=len(keys)) or not torch.equal(keys[order][rows],point_keys):
            raise RuntimeError('Sparse output does not cover input voxels')
        semantic=output.features.argmax(1)[order[rows]]
        remap=torch.tensor([0,2,2,2,2,2,2,2,2,0,0,0,0,1,1,0,1,0,1,1],dtype=torch.uint8,device=points.device)
        return remap[semantic],semantic

def encode_payload(payload):
    """Single copy per representation, exact UTF-8 frame byte count (not estimates)."""
    telemetry=payload['telemetry']
    telemetry['payload_bytes']=0
    for _ in range(10):
        message=json.dumps(payload,separators=(',',':'),allow_nan=False)
        measured=len(message.encode('utf-8'))
        if telemetry['payload_bytes']==measured: return message
        telemetry['payload_bytes']=measured
    raise RuntimeError('Payload size did not converge')
