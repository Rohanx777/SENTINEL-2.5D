# SENTINEL 2.5D — portfolio case study

## Problem

Raw LiDAR scans are difficult to inspect in a browser. This project explores a compact height-grid representation alongside a sampled point-cloud preview.

## Contribution

Integrated an existing pretrained segmentation network with validated scan ingestion, native CUDA C++ components, a compact streaming representation and a Next.js/Deck.gl dashboard. Added explicit runtime modes and measurement records to distinguish real inference from synthetic fixtures.

## Evidence a reviewer can inspect

- Input validation and exact UTF-8 payload-size accounting.
- The boundary between custom pipeline work and the attributed pretrained model.
- Grid representation choices and vertical detail lost compared with a 3D point cloud.
- Sensor-frame visualisation with measured metrics and visible data gaps.
- Benchmark records containing hardware, precision, checkpoint hash, warm-up count and scope.

## Resume wording

**SENTINEL 2.5D | Python, CUDA C++, PyTorch, TypeScript, Next.js, Deck.gl**

- Built a LiDAR research pipeline integrating pretrained semantic segmentation, native CUDA C++ components and compact height-grid visualisation.
- Developed an interactive dashboard comparing sampled point clouds and 2.5D grids, with sensor-frame coordinates and measured processing/payload metrics.
- Recorded 41.41 ms median processing over 30 scans on an RTX 4060 Laptop GPU; documented excluded network/browser time and unverified accuracy.

Use these statements after reviewing and being able to explain the implementation. Do not claim authorship of Cylinder3D, field deployment, validated threat detection, sustained 30 FPS or unmeasured accuracy.

## Interview preparation

1. What do the four input floats mean, and how are malformed scans rejected?
2. What information does a 2.5D grid lose?
3. Which work executes in Python, CUDA C++ and the browser?
4. Why do processing latency and delivered FPS differ?
5. Which parts are upstream research, and what was changed?
6. What evidence is needed for an accuracy claim?

## Readiness

This is an unfinished research project with inspectable engineering work, not a validated perception product. A stronger civilian portfolio release requires labelled-data evaluation, independent setup reproduction, a recorded real-scan demonstration, and resolution of licensing and runtime reliability limitations. The portfolio update includes a recorded synthetic walkthrough and a tested slow-client timeout fix. It does not replace a real-scan demonstration or establish production robustness.
