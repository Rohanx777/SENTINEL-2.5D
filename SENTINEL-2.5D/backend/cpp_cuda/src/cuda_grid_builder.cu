/**
 * cuda_grid_builder.cu
 * --------------------
 * CUDA implementation of 2.5D Cartesian grid compression and temporal smoothing.
 *
 * Algorithm on GPU:
 * 1. Parallel point filter + cell assignment (1 thread per point)
 * 2. Atomic aggregation: count, z_min, z_max, label histogram per cell
 * 3. Cell post-processing (1 thread per cell):
 *    - Check min points threshold (>= 5)
 *    - AI + Geometry fusion heuristic
 *    - 10-frame temporal ring buffer majority vote
 *    - Elevation calculation
 *    - Polygon vertex construction
 * 4. Stream-compact valid cells to host
 */

#include "cuda_grid_builder.cuh"
#include <cmath>
#include <cstring>
#include <vector>

// ── GPU cell state for atomic aggregation ─────────────────────────────────────
struct CellAccumulator {
    int point_count;
    int z_min_int;        // atomicMin with float-as-int (safe for positive/negative floats)
    int z_max_int;        // atomicMax with float-as-int
    int label_votes[3];   // [0: terrain, 1: static, 2: dynamic]
};

// ── GPU temporal history buffer (persists across frames) ───────────────────────
struct TemporalHistory {
    uint8_t history[TOTAL_CELLS][HISTORY_LEN];  // ring buffer per cell
    int head;                                   // current ring index (0..HISTORY_LEN-1)
};

static TemporalHistory* d_history = nullptr;

// Float-to-int conversion preserving order for atomicMin/atomicMax
__device__ inline int float_to_ordered_int(float f) {
    int i = __float_as_int(f);
    return (i >= 0) ? i : (0x7FFFFFFF - i);
}

__device__ inline float ordered_int_to_float(int i) {
    int orig = (i >= 0) ? i : (0x7FFFFFFF - i);
    return __int_as_float(orig);
}

// ── Kernel 1: Reset accumulators ──────────────────────────────────────────────
__global__ void reset_accumulators_kernel(CellAccumulator* accumulators) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_CELLS) return;

    accumulators[idx].point_count = 0;
    accumulators[idx].z_min_int = 0x7FFFFFFF;   // max positive int
    accumulators[idx].z_max_int = -0x7FFFFFFF;  // min int
    accumulators[idx].label_votes[0] = 0;
    accumulators[idx].label_votes[1] = 0;
    accumulators[idx].label_votes[2] = 0;
}

// ── Kernel 2: Point filtering and atomic aggregation ──────────────────────────
__global__ void aggregate_points_kernel(
    const float* __restrict__ points_xyz,
    const uint8_t* __restrict__ labels,
    CellAccumulator* __restrict__ accumulators,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = points_xyz[idx * 3 + 0];
    float y = points_xyz[idx * 3 + 1];
    float z = points_xyz[idx * 3 + 2];
    uint8_t lb = labels[idx];

    // Filter: glass/sky reflections and sensor origin
    float dist_sq = x * x + y * y;
    if (z <= -3.5f || z >= 8.0f || dist_sq <= 0.25f) return;

    // Cartesian index
    int c_idx = (int)floorf((x - GRID_X_MIN) / CELL_SIZE);
    int r_idx = (int)floorf((y - GRID_Y_MIN) / CELL_SIZE);

    if (c_idx < 0 || c_idx >= N_COLS || r_idx < 0 || r_idx >= N_ROWS) return;

    int cell_id = r_idx * N_COLS + c_idx;

    // Atomic updates
    atomicAdd(&accumulators[cell_id].point_count, 1);

    int z_ordered = float_to_ordered_int(z);
    atomicMin(&accumulators[cell_id].z_min_int, z_ordered);
    atomicMax(&accumulators[cell_id].z_max_int, z_ordered);

    if (lb <= 2) {
        atomicAdd(&accumulators[cell_id].label_votes[lb], 1);
    }
}

// ── Kernel 3: Cell post-processing + Temporal Majority Vote ───────────────────
__global__ void process_cells_kernel(
    const CellAccumulator* __restrict__ accumulators,
    TemporalHistory* __restrict__ history,
    CompressedCell* __restrict__ output_cells,
    int* __restrict__ valid_flags,
    int* __restrict__ valid_count
) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= TOTAL_CELLS) return;

    const CellAccumulator& acc = accumulators[cell_id];

    if (acc.point_count < MIN_POINTS_PER_CELL) {
        valid_flags[cell_id] = 0;
        return;
    }

    int ri = cell_id / N_COLS;
    int ci = cell_id % N_COLS;

    float z_min = ordered_int_to_float(acc.z_min_int);
    float z_max = ordered_int_to_float(acc.z_max_int);
    float delta_z = z_max - z_min;

    // ── Find ML dominant vote ──────────────────────────────────────────────────
    int ml_vote = 0;
    int max_votes = acc.label_votes[0];
    if (acc.label_votes[1] > max_votes) {
        ml_vote = 1;
        max_votes = acc.label_votes[1];
    }
    if (acc.label_votes[2] > max_votes) {
        ml_vote = 2;
    }

    // ── AI + Geometry Fusion Heuristic ─────────────────────────────────────────
    int raw_label = 0;  // 0: road, 1: static_obstacle, 2: dynamic_target
    if (ml_vote == 2) {
        raw_label = 2;   // AI detected car/person/cyclist
    } else if (ml_vote == 1 && delta_z > 1.2f) {
        raw_label = 1;   // AI detected building/tree
    } else if (delta_z > 2.0f) {
        raw_label = 1;   // Fallback for tall unregistered objects
    } else if (delta_z > 1.0f && z_max > -0.5f) {
        raw_label = 2;   // Fallback for car-sized objects
    } else {
        raw_label = 0;   // Default to road for flat ground
    }

    // ── Temporal majority vote (10-frame ring buffer) ──────────────────────────
    int head = history->head;
    history->history[cell_id][head] = (uint8_t)raw_label;

    int vote_counts[3] = {0, 0, 0};
    for (int i = 0; i < HISTORY_LEN; i++) {
        uint8_t past_label = history->history[cell_id][i];
        if (past_label <= 2) {
            vote_counts[past_label]++;
        }
    }

    int smoothed_label = 0;
    int best_vote = vote_counts[0];
    if (vote_counts[1] > best_vote) {
        smoothed_label = 1;
        best_vote = vote_counts[1];
    }
    if (vote_counts[2] > best_vote) {
        smoothed_label = 2;
    }

    // ── Compute elevation ──────────────────────────────────────────────────────
    float elev;
    if (smoothed_label == 0) {
        elev = 0.2f;
    } else {
        elev = fmaxf(0.4f, fminf(delta_z, 5.0f));
    }

    // ── Construct polygon vertices ─────────────────────────────────────────────
    float x0 = GRID_X_MIN + ci * CELL_SIZE;
    float y0 = GRID_Y_MIN + ri * CELL_SIZE;
    float x1 = x0 + CELL_SIZE;
    float y1 = y0 + CELL_SIZE;

    // Atomically get output slot
    int slot = atomicAdd(valid_count, 1);

    CompressedCell& cell = output_cells[slot];
    cell.polygon[0][0] = x0; cell.polygon[0][1] = y0;
    cell.polygon[1][0] = x1; cell.polygon[1][1] = y0;
    cell.polygon[2][0] = x1; cell.polygon[2][1] = y1;
    cell.polygon[3][0] = x0; cell.polygon[3][1] = y1;

    cell.elev = elev;
    cell.delta_z = delta_z;
    cell.label = smoothed_label;
    cell.cx = x0 + CELL_SIZE * 0.5f;
    cell.cy = y0 + CELL_SIZE * 0.5f;

    valid_flags[cell_id] = 1;
}

// ── Host lifecycle functions ───────────────────────────────────────────────────
void init_temporal_buffer() {
    if (!d_history) {
        cudaMalloc(&d_history, sizeof(TemporalHistory));
        cudaMemset(d_history, 0, sizeof(TemporalHistory));
    }
}

void free_temporal_buffer() {
    if (d_history) {
        cudaFree(d_history);
        d_history = nullptr;
    }
}

// ── Host launch function ───────────────────────────────────────────────────────
GridResult launch_grid_builder(
    const float* points_xyz,
    const uint8_t* labels,
    int num_points,
    cudaStream_t stream
) {
    init_temporal_buffer();

    // Allocate GPU scratch buffers
    CellAccumulator* d_accumulators = nullptr;
    CompressedCell* d_output_cells = nullptr;
    int* d_valid_flags = nullptr;
    int* d_valid_count = nullptr;

    cudaMalloc(&d_accumulators, TOTAL_CELLS * sizeof(CellAccumulator));
    cudaMalloc(&d_output_cells, TOTAL_CELLS * sizeof(CompressedCell));
    cudaMalloc(&d_valid_flags, TOTAL_CELLS * sizeof(int));
    cudaMalloc(&d_valid_count, sizeof(int));
    cudaMemsetAsync(d_valid_count, 0, sizeof(int), stream);

    // 1. Reset accumulators
    int block_size = 256;
    int grid_cells = (TOTAL_CELLS + block_size - 1) / block_size;
    reset_accumulators_kernel<<<grid_cells, block_size, 0, stream>>>(d_accumulators);

    // 2. Aggregate points
    int grid_points = (num_points + block_size - 1) / block_size;
    aggregate_points_kernel<<<grid_points, block_size, 0, stream>>>(
        points_xyz, labels, d_accumulators, num_points
    );

    // 3. Process cells
    process_cells_kernel<<<grid_cells, block_size, 0, stream>>>(
        d_accumulators, d_history, d_output_cells, d_valid_flags, d_valid_count
    );

    // Advance ring buffer head
    TemporalHistory h_hist;
    cudaMemcpyAsync(&h_hist.head, &d_history->head, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    h_hist.head = (h_hist.head + 1) % HISTORY_LEN;
    cudaMemcpyAsync(&d_history->head, &h_hist.head, sizeof(int), cudaMemcpyHostToDevice, stream);

    // Read back valid count
    int h_valid_count = 0;
    cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Read back valid cells
    GridResult result;
    result.cells.resize(h_valid_count);
    if (h_valid_count > 0) {
        cudaMemcpy(
            result.cells.data(),
            d_output_cells,
            h_valid_count * sizeof(CompressedCell),
            cudaMemcpyDeviceToHost
        );
    }

    result.raw_points_count = num_points;
    result.compressed_cells_count = h_valid_count;
    result.memory_saved_percent = 100.0f * (1.0f - ((float)h_valid_count / (float)(num_points > 0 ? num_points : 1)));

    // Cleanup scratch memory
    cudaFree(d_accumulators);
    cudaFree(d_output_cells);
    cudaFree(d_valid_flags);
    cudaFree(d_valid_count);

    return result;
}
