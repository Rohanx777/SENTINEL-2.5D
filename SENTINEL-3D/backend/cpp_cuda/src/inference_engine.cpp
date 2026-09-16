/**
 * inference_engine.cpp
 * --------------------
 * LibTorch C++ inference engine implementation.
 */

#include "inference_engine.h"
#include <iostream>
#include <stdexcept>

InferenceEngine::InferenceEngine(const std::string& checkpoint_path, const std::string& device)
    : device_(device == "cuda" ? torch::kCUDA : torch::kCPU)
{
    std::cout << "[InferenceEngine] Loading model from: " << checkpoint_path << std::endl;

    try {
        // Load TorchScript model
        model_ = torch::jit::load(checkpoint_path, device_);
        model_.eval();
        std::cout << "[InferenceEngine] Model loaded successfully on " << device << std::endl;
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "Failed to load model. Make sure to export the Python model to TorchScript:\n"
            "  torch.jit.script(model).save('model.pt')\n"
            "Error: " + std::string(e.what())
        );
    }

    // Initialize remap table: 20 SemanticKITTI classes → 3 Kavach classes
    // [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    std::vector<uint8_t> remap_vec = {
        0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1
    };
    remap_table_ = torch::from_blob(
        remap_vec.data(),
        {20},
        torch::TensorOptions().dtype(torch::kUInt8)
    ).clone().to(device_);
}

InferenceEngine::~InferenceEngine() {
    // LibTorch handles cleanup automatically
}

torch::Tensor InferenceEngine::infer(
    const torch::Tensor& pt_fea,
    const torch::Tensor& grid_ind,
    int num_points
) {
    if (num_points == 0) {
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
    }

    torch::NoGradGuard no_grad;

    // Prepare inputs as list of tensors (batch_size=1)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(std::vector<torch::Tensor>{pt_fea});
    inputs.push_back(std::vector<torch::Tensor>{grid_ind.to(torch::kInt32)});
    inputs.push_back(1);  // batch_size

    // Forward pass
    auto output = model_.forward(inputs).toTensor();  // (1, 20, 480, 360, 32)

    // Argmax over class dimension (dim=1)
    auto vox_labels = output[0].argmax(0);  // (480, 360, 32)

    // Map point indices to voxel labels
    auto gx = grid_ind.index({torch::indexing::Slice(), 0}).to(torch::kLong).clamp(0, GRID_SIZE[0] - 1);
    auto gy = grid_ind.index({torch::indexing::Slice(), 1}).to(torch::kLong).clamp(0, GRID_SIZE[1] - 1);
    auto gz = grid_ind.index({torch::indexing::Slice(), 2}).to(torch::kLong).clamp(0, GRID_SIZE[2] - 1);

    // Index into voxel labels
    auto pt_sem = vox_labels.index({gx, gy, gz});  // (N,) ∈ [0, 19]

    // Remap 20 → 3 classes
    auto pt_target = remap_table_.index({pt_sem});  // (N,) ∈ {0, 1, 2}

    return pt_target;
}
