/**
 * cuda_grid_builder.cuh
 * ---------------------
 * CUDA accelerated 2.5D Cartesian grid builder.
 * Replaces Python build_grid() for-loop (~150ms) with GPU kernel (~2ms).
 *
 * Expected speedup: 75×
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// ── Cartesian grid configuration (must match Python config) ───────────────────
constexpr float GRID_X_MIN = -40.0f;
constexpr float GRID_X_MAX = 40.0f;
constexpr float GRID_Y_MIN = -20.0f;
constexpr float GRID_Y_MAX = 20.0f;
constexpr float CELL_SIZE = 2.0f;

constexpr int N_COLS = (int)((GRID_X_MAX - GRID_X_MIN) / CELL_SIZE);  // 40
constexpr int N_ROWS = (int)((GRID_Y_MAX - GRID_Y_MIN) / CELL_SIZE);  // 20
constexpr int TOTAL_CELLS = N_COLS * N_ROWS;                         // 800

constexpr int HISTORY_LEN = 10;  // 10-frame majority vote buffer
constexpr int MIN_POINTS_PER_CELL = 5;

// ── Compressed cell output structure ──────────────────────────────────────────
struct CompressedCell {
    float polygon[4][2];  // 4 vertices [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
    float elev;           // extruded height
    float delta_z;        // max_z - min_z
    int label;            // 0: road, 1: static_obstacle, 2: dynamic_target
    float cx;             // center x
    float cy;             // center y
};

// ── Host result ────────────────────────────────────────────────────────────────
struct GridResult {
    std::vector<CompressedCell> cells;
    int raw_points_count;
    int compressed_cells_count;
    float memory_saved_percent;
};

/**
 * Initialize temporal history buffers on GPU.
 */
void init_temporal_buffer();

/**
 * Free temporal history buffers.
 */
void free_temporal_buffer();

/**
 * Launch 2.5D grid building kernel on GPU.
 *
 * @param points_xyz    (N, 3) float32 device pointer [x, y, z]
 * @param labels        (N,) uint8 device pointer (0=Terrain, 1=Static, 2=Dynamic)
 * @param num_points    Number of points N
 * @param stream        CUDA stream for async execution
 * @return              GridResult with valid compressed cells
 */
GridResult launch_grid_builder(
    const float* points_xyz,
    const uint8_t* labels,
    int num_points,
    cudaStream_t stream = 0
);
