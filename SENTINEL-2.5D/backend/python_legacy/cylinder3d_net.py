"""Official Cylinder3D adapter; see vendor/NOTICE.md. Missing weights are errors.
Features: [delta_rho,delta_phi,delta_z,rho,phi,z,x,y,intensity]; indices: N x 3.
"""
from pathlib import Path
import numpy as np
import torch
GRID_SIZE = np.array([480,360,32], dtype=np.int32)
MAX_BOUND = np.array([50.0,np.pi,2.0], dtype=np.float32)
MIN_BOUND = np.array([0.0,-np.pi,-4.0], dtype=np.float32)
FEA_DIM = 9
N_CLASSES = 20

def cart2polar(xyz):
    return np.stack([np.hypot(xyz[:,0],xyz[:,1]),np.arctan2(xyz[:,1],xyz[:,0]),xyz[:,2]], axis=1)

class CylinderVoxelizer:
    def __init__(self, grid_size=GRID_SIZE, max_bound=MAX_BOUND, min_bound=MIN_BOUND):
        self.grid_size = np.asarray(grid_size)
        self.max_bound = np.asarray(max_bound)
        self.min_bound = np.asarray(min_bound)
        self.intervals = (self.max_bound-self.min_bound)/(self.grid_size-1)
    def __call__(self, points):
        points = np.asarray(points,dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError('Expected finite N x 4 [x,y,z,intensity] points')
        xyz = points[:,:3]
        polar = cart2polar(xyz)
        indices = np.floor((np.clip(polar,self.min_bound,self.max_bound)-self.min_bound)/self.intervals).astype(np.int32)
        indices = np.clip(indices,0,self.grid_size-1)
        centers = (indices.astype(np.float32)+0.5)*self.intervals+self.min_bound
        features = np.concatenate([polar-centers,polar,xyz[:,:2],points[:,3:4]],axis=1)
        return torch.from_numpy(features).float(),torch.from_numpy(indices).int()

def build_model(checkpoint_path,device=torch.device('cuda'),grid_size=(480,360,32)):
    path = Path(checkpoint_path)
    if not path.is_file(): raise FileNotFoundError(f'Trained checkpoint missing: {path}. Supply --weights or use demo mode.')
    if device.type != 'cuda' or not torch.cuda.is_available(): raise RuntimeError('Cylinder3D requires CUDA; demo and replay work on CPU.')
    from vendor.cylinder_fea_generator import cylinder_fea
    from vendor.segmentator_3d_asymm_spconv import Asymm_3d_spconv
    from vendor.cylinder_spconv_3d import cylinder_asym
    model = cylinder_asym(cylinder_fea(grid_size,fea_dim=9,out_pt_fea_dim=256,fea_compre=16),
        Asymm_3d_spconv(grid_size,num_input_features=16,nclasses=20,init_size=32),grid_size)
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    for key in ('model','state_dict','model_state_dict'):
        if key in checkpoint and isinstance(checkpoint[key],dict):
            checkpoint=checkpoint[key]
            break
    checkpoint={k.removeprefix('module.'):v for k,v in checkpoint.items()}
    expected=model.state_dict()
    for key,value in list(checkpoint.items()):
        if key in expected and value.ndim==5:
            if '.upBlock' in key and key.endswith('.conv3.weight') and value.shape[:3]==(3,3,3):
                # Explicit migration of the released legacy cached-rulebook kernel.
                # Inactive planes in the official file contain uninitialized NaNs.
                value=value[:1].contiguous()
            converted=value.permute(4,0,1,2,3).contiguous()
            if converted.shape==expected[key].shape: checkpoint[key]=converted
        if not torch.isfinite(checkpoint[key]).all():
            raise ValueError(f'Nonfinite checkpoint parameter: {key}')
    model.load_state_dict(checkpoint,strict=True)
    return model.to(device).eval()
