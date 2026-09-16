/**
 * inference_engine.h
 * ------------------
 * LibTorch C++ wrapper for Cylinder3D spconv model inference.
 * Replaces Python PyTorch eager execution with optimized C++ runtime.
 */

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <cstdint>

class InferenceEngine {
public:
    /**
     * Initialize inference engine and load Cylinder3D model.
     *
     * @param checkpoint_path   Path to .pt or .pth checkpoint (TorchScript or state_dict)
     * @param device            "cuda" or "cpu"
     */
    InferenceEngine(const std::string& checkpoint_path, const std::string& device = "cuda");
    ~InferenceEngine();

    /**
     * Run inference on voxelized point cloud.
     *
     * @param pt_fea        (N, 9) float32 tensor on GPU — per-point features
     * @param grid_ind      (N, 3) int32 tensor on GPU — voxel indices
     * @param num_points    Number of points N
     * @return              (N,) uint8 tensor on GPU — remapped labels {0, 1, 2}
     */
    torch::Tensor infer(
        const torch::Tensor& pt_fea,
        const torch::Tensor& grid_ind,
        int num_points
    );

    /**
     * Get device (cuda or cpu).
     */
    torch::Device get_device() const { return device_; }

private:
    torch::jit::script::Module model_;
    torch::Device device_;

    // SemanticKITTI 20-class → 3-class remap tensor
    torch::Tensor remap_table_;

    // Grid size [480, 360, 32]
    static constexpr int GRID_SIZE[3] = {480, 360, 32};
};
