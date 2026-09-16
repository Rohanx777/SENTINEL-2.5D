/**
 * python_bindings.cpp
 * -------------------
 * pybind11 wrapper for CUDA kernels.
 * Allows calling optimized CUDA code from Python for incremental migration.
 *
 * Build with: cmake -DBUILD_PYTHON_BINDINGS=ON ..
 *
 * Usage from Python:
 *   import kaavach_cuda_py
 *   result = kaavach_cuda_py.voxelize(points_xyz, points_sig)
 *   grid = kaavach_cuda_py.build_grid(points_xyz, labels)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cuda_voxelizer.cuh"
#include "cuda_grid_builder.cuh"

#include <torch/extension.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// ── Wrapper: CUDA voxelizer ────────────────────────────────────────────────────
py::tuple py_voxelize(
    py::array_t<float> points_xyz,  // (N, 3)
    py::array_t<float> points_sig   // (N,)
) {
    auto xyz_buf = points_xyz.request();
    auto sig_buf = points_sig.request();

    if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3) {
        throw std::runtime_error("points_xyz must be (N, 3)");
    }
    if (sig_buf.ndim != 1) {
        throw std::runtime_error("points_sig must be (N,)");
    }

    int n_points = xyz_buf.shape[0];

    // Upload to GPU
    float* d_xyz = nullptr;
    float* d_sig = nullptr;
    cudaMalloc(&d_xyz, n_points * 3 * sizeof(float));
    cudaMalloc(&d_sig, n_points * sizeof(float));

    cudaMemcpy(d_xyz, xyz_buf.ptr, n_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig_buf.ptr, n_points * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    VoxelizationResult result = launch_voxelizer(d_xyz, d_sig, n_points);
    cudaDeviceSynchronize();

    // Download results
    py::array_t<float> pt_fea({n_points, FEA_DIM});
    py::array_t<int32_t> grid_ind({n_points, 3});

    cudaMemcpy(pt_fea.mutable_data(), result.pt_fea, n_points * FEA_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_ind.mutable_data(), result.grid_ind, n_points * 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_xyz);
    cudaFree(d_sig);
    free_voxelization_result(result);

    return py::make_tuple(pt_fea, grid_ind);
}

// ── Wrapper: CUDA grid builder ─────────────────────────────────────────────────
py::list py_build_grid(
    py::array_t<float> points_xyz,  // (N, 3)
    py::array_t<uint8_t> labels     // (N,)
) {
    auto xyz_buf = points_xyz.request();
    auto lbl_buf = labels.request();

    if (xyz_buf.ndim != 2 || xyz_buf.shape[1] != 3) {
        throw std::runtime_error("points_xyz must be (N, 3)");
    }
    if (lbl_buf.ndim != 1) {
        throw std::runtime_error("labels must be (N,)");
    }

    int n_points = xyz_buf.shape[0];

    // Upload to GPU
    float* d_xyz = nullptr;
    uint8_t* d_labels = nullptr;
    cudaMalloc(&d_xyz, n_points * 3 * sizeof(float));
    cudaMalloc(&d_labels, n_points * sizeof(uint8_t));

    cudaMemcpy(d_xyz, xyz_buf.ptr, n_points * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, lbl_buf.ptr, n_points * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    GridResult result = launch_grid_builder(d_xyz, d_labels, n_points);

    // Convert to Python list of dicts
    py::list cells;
    for (const auto& cell : result.cells) {
        py::dict cell_dict;

        py::list polygon;
        for (int i = 0; i < 4; i++) {
            py::list vertex;
            vertex.append(cell.polygon[i][0]);
            vertex.append(cell.polygon[i][1]);
            polygon.append(vertex);
        }

        cell_dict["polygon"] = polygon;
        cell_dict["elev"] = cell.elev;
        cell_dict["delta_z"] = cell.delta_z;
        cell_dict["label"] = cell.label;
        cell_dict["cx"] = cell.cx;
        cell_dict["cy"] = cell.cy;

        cells.append(cell_dict);
    }

    // Cleanup
    cudaFree(d_xyz);
    cudaFree(d_labels);

    return cells;
}

// ── Module definition ──────────────────────────────────────────────────────────
PYBIND11_MODULE(kaavach_cuda_py, m) {
    m.doc() = "CUDA-accelerated Kavach LiDAR processing kernels";

    m.def("voxelize", &py_voxelize,
          "Cylindrical voxelization on GPU. Returns (pt_fea, grid_ind).",
          py::arg("points_xyz"), py::arg("points_sig"));

    m.def("build_grid", &py_build_grid,
          "2.5D Cartesian grid compression on GPU with temporal smoothing. Returns list of cell dicts.",
          py::arg("points_xyz"), py::arg("labels"));

    m.def("init_temporal_buffer", &init_temporal_buffer,
          "Initialize GPU temporal history buffer (call once at startup).");

    m.def("free_temporal_buffer", &free_temporal_buffer,
          "Free GPU temporal history buffer (call at shutdown).");
}
