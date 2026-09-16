/**
 * cuda_voxelizer.cuh
 * ------------------
 * CUDA accelerated cylindrical voxelization.
 * Replaces Python CylinderVoxelizer with a single kernel launch.
 *
 * Expected speedup: 20× (20ms CPU NumPy → 1ms GPU)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ── Grid configuration (must match Python config) ──────────────────────────────
constexpr int GRID_SIZE_X = 480;
constexpr int GRID_SIZE_Y = 360;
constexpr int GRID_SIZE_Z = 32;

constexpr float MAX_BOUND_RHO = 50.0f;
constexpr float MAX_BOUND_PHI = 3.14159265359f;   // π
constexpr float MAX_BOUND_Z = 2.0f;

constexpr float MIN_BOUND_RHO = 0.0f;
constexpr float MIN_BOUND_PHI = -3.14159265359f;  // -π
constexpr float MIN_BOUND_Z = -4.0f;

constexpr int FEA_DIM = 9;  // per-point feature dimensionality

// ── Voxelization result ────────────────────────────────────────────────────────
struct VoxelizationResult {
    float* pt_fea;       // (N, 9) float32 — per-point features on GPU
    int32_t* grid_ind;   // (N, 3) int32 — voxel indices on GPU
    int num_points;
};

/**
 * Launch cylindrical voxelization on GPU.
 *
 * @param points_xyz    (N, 3) float32 device pointer [x, y, z] in Cartesian
 * @param points_sig    (N,) float32 device pointer — intensity values
 * @param num_points    Number of points N
 * @param stream        CUDA stream for async execution
 * @return              VoxelizationResult with GPU pointers (caller must free)
 */
VoxelizationResult launch_voxelizer(
    const float* points_xyz,
    const float* points_sig,
    int num_points,
    cudaStream_t stream = 0
);

/**
 * Free voxelization result GPU memory.
 */
void free_voxelization_result(VoxelizationResult& result);
