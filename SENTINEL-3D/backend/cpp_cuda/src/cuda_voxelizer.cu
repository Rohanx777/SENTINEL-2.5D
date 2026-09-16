/**
 * cuda_voxelizer.cu
 * -----------------
 * CUDA kernel for cylindrical voxelization.
 * Replaces Python NumPy operations with GPU parallelism.
 */

#include "cuda_voxelizer.cuh"
#include <cmath>

// ── Helper: compute grid intervals ─────────────────────────────────────────────
__device__ inline float3 compute_intervals() {
    return make_float3(
        (MAX_BOUND_RHO - MIN_BOUND_RHO) / (GRID_SIZE_X - 1),
        (MAX_BOUND_PHI - MIN_BOUND_PHI) / (GRID_SIZE_Y - 1),
        (MAX_BOUND_Z - MIN_BOUND_Z) / (GRID_SIZE_Z - 1)
    );
}

// ── Kernel: Cartesian → Polar → Voxel Indices + 9D Features ───────────────────
__global__ void voxelize_kernel(
    const float* __restrict__ points_xyz,  // (N, 3)
    const float* __restrict__ points_sig,  // (N,)
    float* __restrict__ pt_fea,            // (N, 9) output
    int32_t* __restrict__ grid_ind,        // (N, 3) output
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Load Cartesian coordinates
    float x = points_xyz[idx * 3 + 0];
    float y = points_xyz[idx * 3 + 1];
    float z = points_xyz[idx * 3 + 2];
    float intensity = points_sig[idx];

    // ── Cartesian → Cylindrical ───────────────────────────────────────────────
    float rho = sqrtf(x * x + y * y);
    float phi = atan2f(y, x);
    // z remains z

    // ── Clip to bounds ─────────────────────────────────────────────────────────
    rho = fminf(fmaxf(rho, MIN_BOUND_RHO), MAX_BOUND_RHO);
    phi = fminf(fmaxf(phi, MIN_BOUND_PHI), MAX_BOUND_PHI);
    float z_clipped = fminf(fmaxf(z, MIN_BOUND_Z), MAX_BOUND_Z);

    // ── Compute grid indices ──────────────────────────────────────────────────
    float3 intervals = compute_intervals();

    int gx = (int)floorf((rho - MIN_BOUND_RHO) / intervals.x);
    int gy = (int)floorf((phi - MIN_BOUND_PHI) / intervals.y);
    int gz = (int)floorf((z_clipped - MIN_BOUND_Z) / intervals.z);

    // Clamp to grid bounds
    gx = min(max(gx, 0), GRID_SIZE_X - 1);
    gy = min(max(gy, 0), GRID_SIZE_Y - 1);
    gz = min(max(gz, 0), GRID_SIZE_Z - 1);

    // ── Compute voxel center in polar space ───────────────────────────────────
    float vox_center_rho = (gx + 0.5f) * intervals.x + MIN_BOUND_RHO;
    float vox_center_phi = (gy + 0.5f) * intervals.y + MIN_BOUND_PHI;
    float vox_center_z   = (gz + 0.5f) * intervals.z + MIN_BOUND_Z;

    // ── Compute 9D per-point features ─────────────────────────────────────────
    // [Δρ, Δφ, Δz, ρ, φ, z, x_cart, y_cart, intensity]
    float delta_rho = rho - vox_center_rho;
    float delta_phi = phi - vox_center_phi;
    float delta_z   = z_clipped - vox_center_z;

    int fea_base = idx * FEA_DIM;
    pt_fea[fea_base + 0] = delta_rho;
    pt_fea[fea_base + 1] = delta_phi;
    pt_fea[fea_base + 2] = delta_z;
    pt_fea[fea_base + 3] = rho;
    pt_fea[fea_base + 4] = phi;
    pt_fea[fea_base + 5] = z_clipped;
    pt_fea[fea_base + 6] = x;
    pt_fea[fea_base + 7] = y;
    pt_fea[fea_base + 8] = intensity;

    // ── Write grid indices ────────────────────────────────────────────────────
    int ind_base = idx * 3;
    grid_ind[ind_base + 0] = gx;
    grid_ind[ind_base + 1] = gy;
    grid_ind[ind_base + 2] = gz;
}

// ── Host launch function ───────────────────────────────────────────────────────
VoxelizationResult launch_voxelizer(
    const float* points_xyz,
    const float* points_sig,
    int num_points,
    cudaStream_t stream
) {
    VoxelizationResult result;
    result.num_points = num_points;

    // Allocate output buffers
    cudaMalloc(&result.pt_fea, num_points * FEA_DIM * sizeof(float));
    cudaMalloc(&result.grid_ind, num_points * 3 * sizeof(int32_t));

    // Launch kernel
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    voxelize_kernel<<<grid_size, block_size, 0, stream>>>(
        points_xyz, points_sig, result.pt_fea, result.grid_ind, num_points
    );

    return result;
}

void free_voxelization_result(VoxelizationResult& result) {
    if (result.pt_fea) cudaFree(result.pt_fea);
    if (result.grid_ind) cudaFree(result.grid_ind);
    result.pt_fea = nullptr;
    result.grid_ind = nullptr;
}
