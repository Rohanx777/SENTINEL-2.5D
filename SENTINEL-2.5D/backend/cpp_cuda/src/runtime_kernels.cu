// SENTINEL runtime-compiled CUDA C++ kernels. No toolkit download required.
// Inputs and allocations belong to PyTorch; launches use its current stream.
extern "C" __global__ void voxelize(const float* points, float* features, int* indices, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float x=points[4*i],y=points[4*i+1],z=points[4*i+2];
    float r=sqrtf(x*x+y*y),p=atan2f(y,x);
    float dr=50.0f/479.0f,dp=6.283185307179586f/359.0f,dz=6.0f/31.0f;
    int ir=min(479,max(0,(int)floorf(fminf(50.0f,fmaxf(0.0f,r))/dr)));
    int ip=min(359,max(0,(int)floorf((fminf(3.141592653589793f,fmaxf(-3.141592653589793f,p))+3.141592653589793f)/dp)));
    int iz=min(31,max(0,(int)floorf((fminf(2.0f,fmaxf(-4.0f,z))+4.0f)/dz)));
    indices[3*i]=ir;indices[3*i+1]=ip;indices[3*i+2]=iz;
    features[9*i]=r-(ir+0.5f)*dr;
    features[9*i+1]=p-((ip+0.5f)*dp-3.141592653589793f);
    features[9*i+2]=z-((iz+0.5f)*dz-4.0f);
    features[9*i+3]=r;features[9*i+4]=p;features[9*i+5]=z;
    features[9*i+6]=x;features[9*i+7]=y;features[9*i+8]=points[4*i+3];
}

__device__ float atomicMinFloat(float* address,float value) {
    int* p=(int*)address;int old=*p,assumed;
    do { assumed=old;old=atomicCAS(p,assumed,__float_as_int(fminf(value,__int_as_float(assumed)))); } while(assumed!=old);
    return __int_as_float(old);
}
__device__ float atomicMaxFloat(float* address,float value) {
    int* p=(int*)address;int old=*p,assumed;
    do { assumed=old;old=atomicCAS(p,assumed,__float_as_int(fmaxf(value,__int_as_float(assumed)))); } while(assumed!=old);
    return __int_as_float(old);
}
extern "C" __global__ void aggregate(const float* points,const unsigned char* labels,
    float* minima,float* maxima,float* sums,int* counts,int* votes,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    float x=points[4*i],y=points[4*i+1],z=points[4*i+2];
    float r=sqrtf(x*x+y*y);if(z<=-4||z>=12||r<0.5f||r>=100) return;
    int offset,sectors,ring;float step;
    if(r<10) {offset=0;sectors=128;step=0.5f;ring=(int)floorf((r-0.5f)/step);}
    else if(r<30) {offset=2432;sectors=64;step=1;ring=(int)floorf(r-10);}
    else {offset=3712;sectors=32;step=2.5f;ring=(int)floorf((r-30)/step);}
    int sector=min(sectors-1,(int)floorf((atan2f(y,x)+3.141592653589793f)*sectors/6.283185307179586f));
    int cell=offset+ring*sectors+max(sector,0);
    atomicMinFloat(minima+cell,z);atomicMaxFloat(maxima+cell,z);atomicAdd(sums+cell,z);
    atomicAdd(counts+cell,1);int label=labels[i];if(label<3) atomicAdd(votes+3*cell+label,1);
}
