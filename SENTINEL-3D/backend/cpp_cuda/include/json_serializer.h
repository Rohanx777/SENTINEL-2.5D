/**
 * json_serializer.h
 * -----------------
 * Ultra-fast zero-allocation JSON serializer for LiDAR grid payloads.
 * Avoids heavy generic JSON libraries (nlohmann/json) by directly formatting strings.
 */

#pragma once

#include "cuda_grid_builder.cuh"
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

// Label names matching frontend expectations
static const char* LABEL_STRINGS[] = {
    "road",
    "static_obstacle",
    "dynamic_target"
};

/**
 * Serialize a full frame payload to JSON string.
 *
 * Output schema matches the Python backend exactly:
 * {
 *   "header": { "frame_id": int, "timestamp": double, "active_engine": "CUDA_TIER_1" },
 *   "telemetry": { "fps": float, "latency_ms": float, "raw_points_count": int,
 *                  "compressed_cells_count": int, "memory_saved_percent": float },
 *   "grid_data": [
 *     { "polygon": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], "elev": float,
 *       "delta_z": float, "label": string, "cx": float, "cy": float }, ...
 *   ],
 *   "threats": []
 * }
 */
inline std::string serialize_payload(
    int frame_id,
    float latency_ms,
    float fps,
    const GridResult& grid_result
) {
    // Pre-allocate buffer to prevent reallocations (~20-30KB per frame)
    std::string json;
    json.reserve(32768);

    // Get timestamp
    auto now = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();

    // ── Header ─────────────────────────────────────────────────────────────────
    json.append("{\"header\":{");
    json.append("\"frame_id\":").append(std::to_string(frame_id)).append(",");
    json.append("\"timestamp\":").append(std::to_string(timestamp)).append(",");
    json.append("\"active_engine\":\"CUDA_TIER_1\"");
    json.append("},");

    // ── Telemetry ──────────────────────────────────────────────────────────────
    json.append("\"telemetry\":{");
    json.append("\"fps\":").append(std::to_string(fps)).append(",");
    json.append("\"latency_ms\":").append(std::to_string(latency_ms)).append(",");
    json.append("\"raw_points_count\":").append(std::to_string(grid_result.raw_points_count)).append(",");
    json.append("\"compressed_cells_count\":").append(std::to_string(grid_result.compressed_cells_count)).append(",");
    json.append("\"memory_saved_percent\":").append(std::to_string(grid_result.memory_saved_percent));
    json.append("},");

    // ── Grid Data (Wedge polygons) ─────────────────────────────────────────────
    json.append("\"grid_data\":[");

    char buf[256];
    for (size_t i = 0; i < grid_result.cells.size(); i++) {
        const auto& cell = grid_result.cells[i];

        if (i > 0) json.push_back(',');

        const char* label_str = (cell.label >= 0 && cell.label <= 2)
            ? LABEL_STRINGS[cell.label]
            : "unknown";

        snprintf(buf, sizeof(buf),
            "{\"polygon\":[[%.3f,%.3f],[%.3f,%.3f],[%.3f,%.3f],[%.3f,%.3f]],"
            "\"elev\":%.3f,\"delta_z\":%.3f,\"label\":\"%s\",\"cx\":%.2f,\"cy\":%.2f}",
            cell.polygon[0][0], cell.polygon[0][1],
            cell.polygon[1][0], cell.polygon[1][1],
            cell.polygon[2][0], cell.polygon[2][1],
            cell.polygon[3][0], cell.polygon[3][1],
            cell.elev, cell.delta_z, label_str, cell.cx, cell.cy
        );

        json.append(buf);
    }

    json.append("],");

    // ── Threats ────────────────────────────────────────────────────────────────
    json.append("\"threats\":[]}");

    return json;
}
