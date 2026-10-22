import os
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend/python_legacy'))
pytestmark=pytest.mark.skipif(os.environ.get('SENTINEL_TEST_CUDA')!='1',reason='Set SENTINEL_TEST_CUDA=1 to compile and test CUDA C++')

def test_cuda_voxel_and_grid_parity():
    import torch
    from cuda_runtime import CudaKernels
    from cylinder3d_net import CylinderVoxelizer
    from foveated_grid_engine import FoveatedGridEngine
    rng=np.random.default_rng(42)
    pts=np.column_stack([rng.uniform(-45,45,10000),rng.uniform(-45,45,10000),rng.uniform(-3,5,10000),rng.uniform(0,1,10000)]).astype(np.float32)
    labels=rng.integers(0,3,len(pts),dtype=np.uint8)
    kernels=CudaKernels()
    try:
        gpu=torch.from_numpy(pts).cuda()
        fea,indices=kernels.voxelize(gpu)
        ref_fea,ref_indices=CylinderVoxelizer()(pts)
        assert torch.allclose(fea.cpu(),ref_fea,atol=2e-5)
        assert torch.equal(indices.cpu(),ref_indices)
        output=kernels.aggregate(gpu,torch.from_numpy(labels).cuda())
        actual,_=FoveatedGridEngine().project_aggregates(tuple(t.cpu().numpy() for t in output),len(pts))
        expected,_=FoveatedGridEngine().project_point_cloud(pts,labels)
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            assert a['label']==b['label'] and a['point_density']==b['point_density']
            assert abs(a['z_mean']-b['z_mean'])<1e-5
    finally: kernels.close()
