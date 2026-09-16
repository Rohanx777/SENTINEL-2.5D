// app/hooks/useLidarStream.ts
"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export interface CellTuple {
    polygon: [number, number][];
    world_polygon?: [number, number][];
    elev: number;
    delta_z: number;
    z_mean?: number;
    slope_deg?: number;
    drivable?: boolean;
    label: string;
    cx: number;
    cy: number;
    world_cx?: number;
    world_cy?: number;
    r?: number;
    azimuth_deg?: number;
    zone_id?: number;
    zone_name?: string;
    point_density?: number;
    color?: [number, number, number, number];
}

export interface Point3D {
    x: number;
    y: number;
    z: number;
    intensity: number;
    label: string;
    label_id: number;
}

export interface EgoVehicleState {
    pose: {
        x: number;
        y: number;
        z: number;
        yaw_deg: number;
    };
    speed_kmh: number;
    frame_id: number;
    playback_speed?: number;
}

export interface ComparativeDimension {
    dimension: string;
    winner: string;
    description: string;
}

export interface ComparativeStudyData {
    summary: {
        compression_ratio_pct: number;
        bandwidth_reduction_factor: number;
        latency_improvement_ms: number;
    };
    point_cloud_3d: {
        entity_type: string;
        count: number;
        unit: string;
        raw_size_kb: number;
        raw_size_mb: number;
        estimated_tx_time_ms: number;
        client_memory_mb: number;
        z_fidelity: string;
        computation_load: string;
        tactical_feasibility: string;
    };
    grid_25d: {
        entity_type: string;
        count: number;
        unit: string;
        raw_size_kb: number;
        raw_size_mb: number;
        estimated_tx_time_ms: number;
        client_memory_mb: number;
        z_fidelity: string;
        computation_load: string;
        tactical_feasibility: string;
    };
    tradeoff_analysis: ComparativeDimension[];
}

export interface FoveatedZoneMetric {
    zone_id: number;
    name: string;
    range_meters: string;
    resolution: string;
    accuracy: string;
    points_ingested: number;
    cells_generated: number;
    compression_ratio_pct: number;
    threats_detected: number;
    purpose: string;
}

export interface TacticalRadioLink {
    link_name: string;
    tx_time_3d_sec: number;
    tx_time_foveated_ms: number;
    status: string;
}

export interface StratifiedMetricsData {
    total_points: number;
    total_foveated_cells: number;
    overall_compression_pct: number;
    zones: FoveatedZoneMetric[];
    tactical_radio_simulation: {
        cbr_64kbps: TacticalRadioLink;
        sdr_256kbps: TacticalRadioLink;
        link16_1200kbps: TacticalRadioLink;
    };
}

export interface TelemetryMetrics {
    engine: "CUDA GPU TIER 1" | "CPU OPENMP FALLBACK";
    fps: number;
    latencyMs: number;
    memorySavedPct: number;
    speedKmh?: number;
    playbackSpeed?: number;
}

export interface ThreatObject {
    id?: string;
    type: string;
    distance_m: number;
    coordinates: [number, number];
    world_coordinates?: [number, number];
    velocity_mps?: [number, number];
    relative_speed_kmh?: number;
    heading_deg?: number;
    time_to_collision_sec?: number;
    threat_level?: "CRITICAL" | "HIGH" | "MODERATE" | string;
    urgency_status?: string;
    depth_m?: number;
}

export interface StreamFrame {
    cells: CellTuple[];
    polarCells: CellTuple[];
    linearCells: CellTuple[];
    pointCloud3D: Point3D[];
    comparativeStudy: ComparativeStudyData | null;
    stratifiedMetrics: StratifiedMetricsData | null;
    egoVehicle: EgoVehicleState | null;
    telemetry: TelemetryMetrics;
    threats: ThreatObject[];
}

interface BackendPayload {
    header?: {
        frame_id: number;
        timestamp: number;
        active_engine: string;
        foveated_zones_active?: number;
        representation_modes?: string[];
    };
    telemetry?: {
        fps: number;
        latency_ms?: number;
        latencyMs?: number;
        raw_points_count?: number;
        compressed_cells_count?: number;
        memory_saved_percent?: number;
        memorySavedPct?: number;
        speed_kmh?: number;
        playback_speed?: number;
    };
    grid_data?: any[];
    foveated_grid?: any[];
    polar_grid?: any[];
    linear_grid?: any[];
    pointcloud_3d?: Point3D[];
    comparative_study?: ComparativeStudyData;
    stratified_metrics?: StratifiedMetricsData;
    ego_vehicle?: EgoVehicleState;
    threats?: ThreatObject[];
}

const LABEL_ID_MAP: Record<number, string> = {
    0: "road",
    1: "static_obstacle",
    2: "dynamic_target",
    3: "negative_obstacle",
    4: "curb",
};

function resolveLabel(raw: number | string): string {
    if (typeof raw === "string") return raw;
    return LABEL_ID_MAP[raw] ?? "unknown";
}

function normalizeEngine(tier: string | undefined): TelemetryMetrics["engine"] {
    if (!tier) return "CUDA GPU TIER 1";
    const t = tier.toUpperCase();
    if (t.includes("CUDA") || t.includes("TIER_1") || t.includes("TIER 1") || t.includes("GPU"))
        return "CUDA GPU TIER 1";
    return "CPU OPENMP FALLBACK";
}

function normalizeCellList(rawList: any[] | undefined): CellTuple[] {
    if (!rawList || !rawList.length) return [];
    const first = rawList[0];

    if (first && typeof first === "object" && !Array.isArray(first) && "polygon" in first) {
        return rawList;
    }

    if (Array.isArray(first)) {
        return rawList.map((item: any) => {
            if (Array.isArray(item)) {
                const [x, y, zMax, deltaZ, label] = item;
                const half = 1.0;
                return {
                    polygon: [
                        [x - half, y - half],
                        [x + half, y - half],
                        [x + half, y + half],
                        [x - half, y + half],
                    ],
                    elev: Math.max(0.2, zMax || 0.5),
                    delta_z: deltaZ || 0.1,
                    label: resolveLabel(label),
                    cx: x,
                    cy: y,
                };
            }
            return item;
        });
    }

    return rawList;
}

function adaptPayload(raw: BackendPayload): StreamFrame | null {
    const foveatedCells = normalizeCellList(raw.foveated_grid || raw.polar_grid || raw.grid_data);
    const linearCells = normalizeCellList(raw.linear_grid);
    const defaultCells = foveatedCells.length > 0 ? foveatedCells : linearCells;

    const t = raw.telemetry;
    const header = raw.header;

    const telemetry: TelemetryMetrics = {
        engine: normalizeEngine(header?.active_engine),
        fps: t?.fps ?? 30,
        latencyMs: t?.latency_ms ?? t?.latencyMs ?? 24.5,
        memorySavedPct: t?.memory_saved_percent ?? t?.memorySavedPct ?? 99.2,
        speedKmh: t?.speed_kmh ?? raw.ego_vehicle?.speed_kmh ?? 36.0,
        playbackSpeed: t?.playback_speed ?? 2.0,
    };

    return {
        cells: defaultCells,
        polarCells: foveatedCells,
        linearCells,
        pointCloud3D: raw.pointcloud_3d || [],
        comparativeStudy: raw.comparative_study || null,
        stratifiedMetrics: raw.stratified_metrics || null,
        egoVehicle: raw.ego_vehicle || null,
        telemetry,
        threats: raw.threats ?? [],
    };
}

export function useLidarStream() {
    const [connected, setConnected] = useState(false);
    const [playbackSpeed, setLocalSpeed] = useState(2.0);
    const [frame, setFrame] = useState<StreamFrame>({
        cells: [],
        polarCells: [],
        linearCells: [],
        pointCloud3D: [],
        comparativeStudy: null,
        stratifiedMetrics: null,
        egoVehicle: null,
        telemetry: {
            engine: "CUDA GPU TIER 1",
            fps: 30,
            latencyMs: 24.5,
            memorySavedPct: 99.2,
            speedKmh: 36.0,
            playbackSpeed: 2.0,
        },
        threats: [],
    });

    const wsRef = useRef<WebSocket | null>(null);
    const latestFrameRef = useRef<StreamFrame | null>(null);
    const hasPendingUpdate = useRef(false);

    const setPlaybackSpeed = useCallback((speed: number) => {
        setLocalSpeed(speed);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "set_speed", speed }));
        }
    }, []);

    useEffect(() => {
        let isCancelled = false;
        let reconnectTimer: any = null;
        let animFrameId: number | null = null;

        // RAF loop to throttle updates smoothly
        function updateLoop() {
            if (isCancelled) return;
            if (hasPendingUpdate.current && latestFrameRef.current) {
                setFrame(latestFrameRef.current);
                hasPendingUpdate.current = false;
            }
            animFrameId = requestAnimationFrame(updateLoop);
        }
        animFrameId = requestAnimationFrame(updateLoop);

        function connect() {
            if (isCancelled) return;

            try {
                const wsUrl = "ws://127.0.0.1:8005/ws/stream_map";
                console.log("[SENTINEL-3D] Connecting to:", wsUrl);
                const ws = new WebSocket(wsUrl);
                wsRef.current = ws;

                ws.onopen = () => {
                    if (isCancelled) return;
                    console.log("[SENTINEL-3D] Connected to stream!");
                    setConnected(true);
                    // Sync initial speed
                    ws.send(JSON.stringify({ action: "set_speed", speed: playbackSpeed }));
                };

                ws.onmessage = (event: MessageEvent) => {
                    if (isCancelled) return;
                    try {
                        const raw: BackendPayload =
                            typeof event.data === "string"
                                ? JSON.parse(event.data)
                                : JSON.parse(new TextDecoder().decode(event.data as ArrayBuffer));

                        const adapted = adaptPayload(raw);
                        if (adapted) {
                            latestFrameRef.current = adapted;
                            hasPendingUpdate.current = true;
                        }
                    } catch (err) {
                        console.error("[SENTINEL-3D] Parse error:", err);
                    }
                };

                ws.onclose = () => {
                    if (isCancelled) return;
                    console.log("[SENTINEL-3D] Stream disconnected. Reconnecting...");
                    setConnected(false);
                    reconnectTimer = setTimeout(connect, 1500);
                };

                ws.onerror = (err) => {
                    console.warn("[SENTINEL-3D] WebSocket error:", err);
                    ws.close();
                };
            } catch (err) {
                console.error("[SENTINEL-3D] Connection failed:", err);
                if (!isCancelled) {
                    reconnectTimer = setTimeout(connect, 1500);
                }
            }
        }

        connect();

        return () => {
            isCancelled = true;
            if (animFrameId) cancelAnimationFrame(animFrameId);
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (wsRef.current) wsRef.current.close();
        };
    }, [playbackSpeed]);

    return {
        connected,
        playbackSpeed,
        setPlaybackSpeed,
        ...frame,
    };
}
