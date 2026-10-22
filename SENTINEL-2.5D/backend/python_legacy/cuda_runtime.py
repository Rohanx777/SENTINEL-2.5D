"""Launch CUDA C++ kernels via NVRTC and the CUDA driver, on PyTorch's stream.
No downloads, host compiler, or global CUDA installation changes. Tested on Windows.
"""
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import torch

class CudaKernels:
    def __init__(self):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
        torch.cuda.init()
        torch.empty(1,device='cuda') # Make the primary context current.
        if os.name=='nt':
            folder=Path(torch.__file__).parent/'lib'
            self.dll_dir=os.add_dll_directory(str(folder))
            path=next(folder.glob('nvrtc64_*.dll'))
            self.nvrtc=C.CDLL(str(path)); self.driver=C.WinDLL('nvcuda.dll')
        else:
            path=ctypes.util.find_library('nvrtc')
            if not path: raise RuntimeError('NVRTC library missing')
            self.nvrtc=C.CDLL(path);self.driver=C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        self.nvrtc.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
        for name in ('nvrtcGetPTXSize','nvrtcGetProgramLogSize'):
            getattr(self.nvrtc,name).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)]
        for name in ('nvrtcGetPTX','nvrtcGetProgramLog'):
            getattr(self.nvrtc,name).argtypes=[C.c_void_p,C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)]
        self.driver.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.driver.cuModuleUnload.argtypes=[C.c_void_p]
        self.driver.cuLaunchKernel.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.c_void_p,C.c_void_p]
        program=C.c_void_p()
        source=(Path(__file__).resolve().parents[1]/'cpp_cuda/src/runtime_kernels.cu').read_bytes()
        self.check(self.nvrtc.nvrtcCreateProgram(C.byref(program),source,b'sentinel.cu',0,None,None),'create NVRTC program')
        major,minor=torch.cuda.get_device_capability()
        opts=(C.c_char_p*2)(b'--std=c++14',f'--gpu-architecture=compute_{major}{minor}'.encode())
        result=self.nvrtc.nvrtcCompileProgram(program,2,opts)
        size=C.c_size_t()
        try:
            if result:
                self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size));log=C.create_string_buffer(size.value)
                self.nvrtc.nvrtcGetProgramLog(program,log)
                raise RuntimeError(log.value.decode())
            self.check(self.nvrtc.nvrtcGetPTXSize(program,C.byref(size)),'PTX size')
            ptx=C.create_string_buffer(size.value)
            self.check(self.nvrtc.nvrtcGetPTX(program,ptx),'PTX output')
        finally: self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        self.module=C.c_void_p();self.check(self.driver.cuModuleLoadData(C.byref(self.module),ptx),'load CUDA module')
        self.functions={}
        for name in ('voxelize','aggregate'):
            fn=C.c_void_p();self.check(self.driver.cuModuleGetFunction(C.byref(fn),self.module,name.encode()),name)
            self.functions[name]=fn

    @staticmethod
    def check(result,operation):
        if result: raise RuntimeError(f'{operation}: CUDA/NVRTC error {result}')

    def launch(self,name,tensors,n):
        if not n: return
        values=[C.c_void_p(t.data_ptr()) for t in tensors]+[C.c_int(n)]
        pointers=(C.c_void_p*len(values))(*(C.cast(C.byref(v),C.c_void_p) for v in values))
        self.check(self.driver.cuLaunchKernel(self.functions[name],(n+255)//256,1,1,256,1,1,0,
            C.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None),name)

    def voxelize(self,points):
        points=points.contiguous()
        features=torch.empty((len(points),9),dtype=torch.float32,device=points.device)
        indices=torch.empty((len(points),3),dtype=torch.int32,device=points.device)
        self.launch('voxelize',[points,features,indices],len(points))
        return features,indices

    def aggregate(self,points,labels):
        device=points.device
        minima=torch.full((4608,),float('inf'),device=device)
        maxima=torch.full((4608,),float('-inf'),device=device)
        sums=torch.zeros(4608,device=device)
        counts=torch.zeros(4608,device=device,dtype=torch.int32)
        votes=torch.zeros((4608,3),device=device,dtype=torch.int32)
        self.launch('aggregate',[points,labels.contiguous(),minima,maxima,sums,counts,votes],len(points))
        return minima,maxima,sums,counts,votes

    def close(self):
        if getattr(self,'module',None):
            torch.cuda.synchronize();self.driver.cuModuleUnload(self.module);self.module=None
