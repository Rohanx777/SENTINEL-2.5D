Cylinder3D source by Xinge Zhu and contributors, Apache-2.0.
Source: https://github.com/xinge008/Cylinder3D
Commit: 30a0abb2ca4c657a821a5e9a343934b0789b2365
Adaptations: spconv 2.x replace_feature API; deterministic native PyTorch voxel max aggregation instead of torch_scatter; removed unused imports. Architecture and parameter names retained.

Additional adaptations: separate indice-cache keys for incompatible kernel shapes; optional sparse-logit output; legacy UpBlock conv3 migration to the active (1,3,3) checkpoint plane. The original checkpoint contains non-finite values in inactive planes. Full parity with the original upstream runtime and published accuracy have not been established. The wrapper also includes a separate experimental precision adaptation; this is not part of the authors original implementation.
