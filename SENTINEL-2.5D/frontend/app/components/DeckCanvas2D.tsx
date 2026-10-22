// app/components/DeckCanvas2D.tsx
"use client";

import React, { useMemo, useState, useEffect } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer, PathLayer, ScatterplotLayer, LineLayer } from "@deck.gl/layers";
import { OrbitView, COORDINATE_SYSTEM } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import { Layers, Box, Compass, Sparkles, Navigation } from "lucide-react";
import type { CellTuple, Point3D, EgoVehicleState, ThreatObject } from "../hooks/useLidarStream";

// ─── Label → colour (RGBA) ────────────────────────────────────────────────────
const LABEL_COLORS: Record<string, [number, number, number, number]> = {
    road: [91, 163, 143, 230],
    static_obstacle: [211, 147, 126, 255],
    dynamic_target: [225, 188, 107, 255],
    negative_obstacle: [167, 151, 209, 255],
    pothole: [167, 151, 209, 255],
    curb: [127, 188, 209, 255],
    threat: [211, 147, 126, 255],
    unknown: [118, 132, 153, 190],
};

function getColor(label: string): [number, number, number, number] {
    const key = (label || "").toLowerCase().replace(/\s+/g, "_");
    return (
        LABEL_COLORS[key] ??
        LABEL_COLORS[Object.keys(LABEL_COLORS).find((k) => key.includes(k)) ?? ""] ??
        LABEL_COLORS.unknown
    );
}

interface WedgeCell {
    polygon: [number, number][];
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

interface TooltipInfo {
    x: number;
    y: number;
    cell?: WedgeCell;
    point?: Point3D;
    threat?: ThreatObject;
}

export type GridType = "polar" | "linear";
export type VisualizationMode = "grid_25d" | "pointcloud_3d" | "split_comparison";
export type CoordinateFrame = "sensor" | "world";

interface Props {
    cells: CellTuple[];
    linearCells?: CellTuple[];
    pointCloud3D?: Point3D[];
    threats?: ThreatObject[];
    egoVehicle?: EgoVehicleState | null;
    gridType?: GridType;
    visMode?: VisualizationMode;
    coordinateFrame?: CoordinateFrame;
}

// ─── Camera Views ─────────────────────────────────────────────────────────────
const INITIAL_VIEW_STATE = {
    target: [0, 0, 0] as [number, number, number],
    rotationX: 48,
    rotationOrbit: -25,
    zoom: 2.7,
    minZoom: 0.5,
    maxZoom: 8,
};

// Helper to generate 360° circle paths for range rings
function createRangeRing(radius: number, points: number = 72): [number, number, number][] {
    const ring: [number, number, number][] = [];
    for (let i = 0; i <= points; i++) {
        const theta = (i / points) * 2 * Math.PI;
        ring.push([radius * Math.cos(theta), radius * Math.sin(theta), 0.05]);
    }
    return ring;
}

// Helper to generate radial azimuth rays
function createRadialRay(angleDeg: number, maxRadius: number = 100): [number, number, number][] {
    const rad = (angleDeg * Math.PI) / 180;
    return [
        [0, 0, 0.05],
        [maxRadius * Math.cos(rad), maxRadius * Math.sin(rad), 0.05],
    ];
}

// Helper to generate Cartesian grid lines for Linear representation
function createCartesianGridLines(
    xMin = -50, xMax = 50, yMin = -30, yMax = 30, step = 10
): [number, number, number][][] {
    const lines: [number, number, number][][] = [];
    for (let x = xMin; x <= xMax; x += step) {
        lines.push([[x, yMin, 0.05], [x, yMax, 0.05]]);
    }
    for (let y = yMin; y <= yMax; y += step) {
        lines.push([[xMin, y, 0.05], [xMax, y, 0.05]]);
    }
    return lines;
}

export default function DeckCanvas2D({
    cells,
    linearCells = [],
    pointCloud3D = [],
    threats = [],
    egoVehicle = null,
    gridType = "polar",
    visMode = "grid_25d",
    coordinateFrame = "sensor",
}: Props) {
    const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);
    const [cameraPreset, setCameraPreset] = useState("perspective");
    const [cameraViews, setCameraViews] = useState<Record<string, any>>({
        'main-pane': INITIAL_VIEW_STATE,
        'left-pane': {...INITIAL_VIEW_STATE, zoom: 2},
        'right-pane': {...INITIAL_VIEW_STATE, zoom: 2},
    });
    const selectCamera = (preset: string) => {
        setCameraPreset(preset);
        const camera = preset === 'top'
            ? {...INITIAL_VIEW_STATE, rotationX: 0, rotationOrbit: 0}
            : INITIAL_VIEW_STATE;
        setCameraViews({'main-pane': camera, 'left-pane': {...camera, zoom: 2}, 'right-pane': {...camera, zoom: 2}});
        setTooltip(null);
    };

    // Reset tooltip on mode or coordinate frame switch
    useEffect(() => {
        setTooltip(null);
    }, [gridType, visMode, coordinateFrame]);

    // Select active cell dataset based on gridType (Polar vs Linear)
    const activeRawCells = useMemo(() => {
        return gridType === "linear" && linearCells.length > 0 ? linearCells : cells;
    }, [gridType, linearCells, cells]);

    // Normalize incoming cells
    const normalizedCells = useMemo<WedgeCell[]>(() => {
        if (!activeRawCells || !activeRawCells.length) return [];
        const isWorld = coordinateFrame === "world";

        return activeRawCells.map((c: any) => {
            const cx = isWorld ? (c.world_cx ?? c.cx ?? 0) : (c.cx ?? 0);
            const cy = isWorld ? (c.world_cy ?? c.cy ?? 0) : (c.cy ?? 0);
            const r = c.r ?? Math.sqrt((c.cx ?? 0) ** 2 + (c.cy ?? 0) ** 2);
            const azimuth_deg = c.azimuth_deg ?? ((Math.atan2(c.cy ?? 0, c.cx ?? 0) * 180) / Math.PI + 360) % 360;

            const polygon = isWorld && c.world_polygon
                ? c.world_polygon
                : c.polygon || [
                    [cx - 1, cy - 1],
                    [cx + 1, cy - 1],
                    [cx + 1, cy + 1],
                    [cx - 1, cy + 1],
                ];

            return {
                polygon,
                elev: Math.max(0, c.elev ?? 0),
                delta_z: c.delta_z ?? 0.1,
                z_mean: c.z_mean,
                slope_deg: c.slope_deg,
                drivable: c.drivable,
                label: c.label ?? "road",
                cx: isWorld ? (c.world_cx ?? cx) : cx,
                cy: isWorld ? (c.world_cy ?? cy) : cy,
                world_cx: c.world_cx,
                world_cy: c.world_cy,
                r: Number(r.toFixed(1)),
                azimuth_deg: Number(azimuth_deg.toFixed(1)),
                zone_id: c.zone_id,
                zone_name: c.zone_name,
                point_density: c.point_density,
                color: getColor(c.label ?? "road"),
            };
        });
    }, [activeRawCells, coordinateFrame]);

    // Foveated Zone Ring Boundaries (10m Safety Core, 30m Tactical Corridor, 60m Perimeter)
    const foveatedZoneRings = useMemo(() => {
        return [
            {
                path: createRangeRing(10.0),
                radius: 10.0,
                color: [143, 174, 195, 65],
                width: 1.0,
                name: "Zone 0: Safety Core (0-10m)",
            },
            {
                path: createRangeRing(30.0),
                radius: 30.0,
                color: [143, 174, 195, 55],
                width: 1.0,
                name: "Zone 1: Tactical Corridor (10-30m)",
            },
            {
                path: createRangeRing(60.0),
                radius: 60.0,
                color: [143, 174, 195, 45],
                width: 1.0,
                name: "Zone 2: Perimeter Surveillance (30-100m)",
            },
        ];
    }, []);

    // Polar range rings & azimuth guidelines
    const polarRingsData = useMemo(() => {
        const radii = [5, 10, 20, 30, 45, 60];
        return radii.map((r) => ({
            path: createRangeRing(r),
            radius: r,
        }));
    }, []);

    const azimuthRaysData = useMemo(() => {
        const angles = [0, 45, 90, 135, 180, 225, 270, 315];
        return angles.map((deg) => ({
            path: createRadialRay(deg, 65),
            angle: deg,
        }));
    }, []);

    // Cartesian Orthogonal Grid Lines (for Linear mode)
    const cartesianGridData = useMemo(() => {
        const lines = createCartesianGridLines(-50, 50, -30, 30, 10);
        return lines.map((path) => ({ path }));
    }, []);

    // 3D Point Cloud sample data
    const pointCloudData = useMemo(() => {
        if (!pointCloud3D || !pointCloud3D.length) return [];
        return pointCloud3D.map((pt) => ({
            position: [pt.x, pt.y, pt.z] as [number, number, number],
            color: getColor(pt.label),
            label: pt.label,
            intensity: pt.intensity,
            original: pt,
        }));
    }, [pointCloud3D]);

    // Threat Velocity Vectors (for dynamic kinematic display)
    const threatVectorsData = useMemo(() => {
        if (!threats || !threats.length) return [];
        const isWorld = coordinateFrame === "world";

        return threats.map((th) => {
            const [x0, y0] = isWorld && th.world_coordinates ? th.world_coordinates : th.coordinates;
            const [vx, vy] = th.velocity_mps ?? [-2.0, 0.0];
            const x1 = x0 + vx * 1.5;
            const y1 = y0 + vy * 1.5;
            return {
                sourcePosition: [x0, y0, 1.2] as [number, number, number],
                targetPosition: [x1, y1, 1.2] as [number, number, number],
                threat: th,
            };
        });
    }, [threats, coordinateFrame]);

    // Ego vehicle marker position
    const egoMarkerData = useMemo(() => {
        if (coordinateFrame === "world" && egoVehicle) {
            return [{
                position: [egoVehicle.pose.x, egoVehicle.pose.y, 0.4] as [number, number, number],
                yaw_deg: egoVehicle.pose.yaw_deg,
                speed: egoVehicle.speed_kmh,
            }];
        }
        return [{
            position: [0, 0, 0.3] as [number, number, number],
            yaw_deg: 0,
            speed: egoVehicle?.speed_kmh ?? 36.0,
        }];
    }, [coordinateFrame, egoVehicle]);

    // DeckGL Views Configuration (Split vs Single Viewport)
    const views = useMemo(() => {
        if (visMode === "split_comparison") {
            return [
                new OrbitView({
                    id: "left-pane",
                    x: "0%",
                    y: "0%",
                    width: "50%",
                    height: "100%",
                    orbitAxis: "Z",
                    controller: true,
                }),
                new OrbitView({
                    id: "right-pane",
                    x: "50%",
                    y: "0%",
                    width: "50%",
                    height: "100%",
                    orbitAxis: "Z",
                    controller: true,
                }),
            ];
        }
        return [
            new OrbitView({
                id: "main-pane",
                x: "0%",
                y: "0%",
                width: "100%",
                height: "100%",
                orbitAxis: "Z",
                controller: true,
            }),
        ];
    }, [visMode]);

    // Build layers with strictly unique, mode-isolated IDs to prevent buffer overlapping
    const layers = useMemo(() => {
        const layerList: any[] = [];
        const isPolar = gridType === "polar";
        const is3DMode = visMode === "pointcloud_3d";
        const isSplitMode = visMode === "split_comparison";
        const isGridMode = visMode === "grid_25d" || isSplitMode;

        const leftViewId = isSplitMode ? "left-pane" : undefined;
        const rightViewId = isSplitMode ? "right-pane" : undefined;

        // ── 1. Grid Guidelines & Foveated Zone Boundaries (2.5D Mode / Split Left) ──
        if (isGridMode) {
            if (isPolar && coordinateFrame === "sensor") {
                layerList.push(
                    new PathLayer({
                        widthUnits: "pixels",
                        id: `polar-range-rings-${gridType}-${visMode}-${coordinateFrame}`,
                        data: polarRingsData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getPath: (d: any) => d.path,
                        getColor: [143, 174, 195, 28],
                        getWidth: 1.0,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    }),
                    new PathLayer({
                        widthUnits: "pixels",
                        id: `polar-azimuth-rays-${gridType}-${visMode}-${coordinateFrame}`,
                        data: azimuthRaysData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getPath: (d: any) => d.path,
                        getColor: [143, 174, 195, 18],
                        getWidth: 1.0,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    }),
                    new PathLayer({
                        widthUnits: "pixels",
                        id: `foveated-zone-rings-${gridType}-${visMode}-${coordinateFrame}`,
                        data: foveatedZoneRings,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getPath: (d: any) => d.path,
                        getColor: (d: any) => d.color,
                        getWidth: (d: any) => d.width,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    })
                );
            } else if (!isPolar && coordinateFrame === "sensor") {
                layerList.push(
                    new PathLayer({
                        widthUnits: "pixels",
                        id: `cartesian-grid-lines-${gridType}-${visMode}-${coordinateFrame}`,
                        data: cartesianGridData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getPath: (d: any) => d.path,
                        getColor: [59, 130, 246, 50],
                        getWidth: 1.0,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    })
                );
            }
        }

        // ── 2. Ego Vehicle Origin Markers ──
        if (isGridMode) {
            layerList.push(
                new ScatterplotLayer({
                    id: `ego-marker-grid-${gridType}-${visMode}-${coordinateFrame}`,
                    data: egoMarkerData,
                    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                    getPosition: (d: any) => d.position,
                    getRadius: coordinateFrame === "world" ? 2.2 : 1.6,
                    getFillColor: [6, 182, 212, 255],
                    getLineColor: [255, 255, 255, 255],
                    lineWidthMinPixels: 2,
                    stroked: true,
                    pickable: false,
                    ...(leftViewId ? { viewId: leftViewId } : {}),
                })
            );
        }

        if (is3DMode || isSplitMode) {
            layerList.push(
                new ScatterplotLayer({
                    id: `ego-marker-3d-${visMode}-${coordinateFrame}`,
                    data: egoMarkerData,
                    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                    getPosition: (d: any) => d.position,
                    getRadius: coordinateFrame === "world" ? 2.2 : 1.6,
                    getFillColor: [168, 85, 247, 255],
                    getLineColor: [255, 255, 255, 255],
                    lineWidthMinPixels: 2,
                    stroked: true,
                    pickable: false,
                    ...(rightViewId ? { viewId: rightViewId } : {}),
                })
            );
        }

        // ── 3. Extruded 2.5D Grid Wedges / Boxes (ONLY in 2.5D or Split Left) ──
        if (isGridMode) {
            layerList.push(
                new PolygonLayer<WedgeCell>({
                    id: `lidar-grid-${gridType}-${visMode}-${coordinateFrame}`,
                    data: normalizedCells,
                    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                    getPolygon: (d) => d.polygon,
                    getElevation: (d) => d.elev,
                    getFillColor: (d) => d.color || [34, 197, 94, 200],
                    getLineColor: [25, 39, 51, 90],
                    lineWidthMinPixels: 0.5,
                    stroked: true,
                    material: {ambient: 0.5, diffuse: 0.65, shininess: 18, specularColor: [80, 90, 100]},
                    pickable: true,
                    extruded: true,
                    elevationScale: 1,
                    updateTriggers: {
                        getElevation: normalizedCells,
                        getFillColor: normalizedCells,
                        getPolygon: normalizedCells,
                    },
                    onHover: (info: PickingInfo<WedgeCell>) => {
                        if (info.object) {
                            setTooltip({ x: info.x, y: info.y, cell: info.object });
                        } else {
                            setTooltip(null);
                        }
                    },
                    ...(leftViewId ? { viewId: leftViewId } : {}),
                })
            );
        }

        // ── 4. 3D Point Cloud Layer (ONLY in 3D Mode or Split Right) ──
        if (is3DMode || isSplitMode) {
            layerList.push(
                new ScatterplotLayer({
                    id: `lidar-3d-pointcloud-${visMode}-${coordinateFrame}`,
                    data: pointCloudData,
                    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                    getPosition: (d: any) => d.position,
                    getRadius: 0.09,
                    radiusMinPixels: 1,
                    radiusMaxPixels: 3,
                    getFillColor: (d: any) => d.color,
                    pickable: true,
                    updateTriggers: {
                        getPosition: pointCloudData,
                        getFillColor: pointCloudData,
                    },
                    onHover: (info: PickingInfo<any>) => {
                        if (info.object) {
                            setTooltip({ x: info.x, y: info.y, point: info.object.original });
                        } else {
                            setTooltip(null);
                        }
                    },
                    ...(rightViewId ? { viewId: rightViewId } : {}),
                })
            );
        }

        // ── 5. Dynamic Threat Velocity Vectors ──
        if (threatVectorsData.length > 0) {
            if (isGridMode) {
                layerList.push(
                    new LineLayer({
                        id: `threat-velocity-vectors-grid-${gridType}-${visMode}-${coordinateFrame}`,
                        data: threatVectorsData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getSourcePosition: (d: any) => d.sourcePosition,
                        getTargetPosition: (d: any) => d.targetPosition,
                        getColor: [239, 68, 68, 255],
                        getWidth: 3.0,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    }),
                    new ScatterplotLayer({
                        id: `threat-marker-heads-grid-${gridType}-${visMode}-${coordinateFrame}`,
                        data: threatVectorsData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getPosition: (d: any) => d.sourcePosition,
                        getRadius: 1.2,
                        getFillColor: [239, 68, 68, 200],
                        getLineColor: [255, 255, 255, 255],
                        lineWidthMinPixels: 1.5,
                        stroked: true,
                        pickable: false,
                        ...(leftViewId ? { viewId: leftViewId } : {}),
                    })
                );
            }

            if (is3DMode || isSplitMode) {
                layerList.push(
                    new LineLayer({
                        id: `threat-velocity-vectors-3d-${visMode}-${coordinateFrame}`,
                        data: threatVectorsData,
                        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                        getSourcePosition: (d: any) => d.sourcePosition,
                        getTargetPosition: (d: any) => d.targetPosition,
                        getColor: [239, 68, 68, 255],
                        getWidth: 3.0,
                        widthMinPixels: 1,
                        pickable: false,
                        ...(rightViewId ? { viewId: rightViewId } : {}),
                    })
                );
            }
        }

        return layerList;
    }, [
        gridType,
        visMode,
        coordinateFrame,
        polarRingsData,
        azimuthRaysData,
        foveatedZoneRings,
        cartesianGridData,
        egoMarkerData,
        normalizedCells,
        pointCloudData,
        threatVectorsData,
    ]);

    return (
        <div className="relative w-full h-full bg-[#101923] overflow-hidden select-none">
            {/* Tactical Radar Radial Background Glow */}
            <div
                className="absolute inset-0 pointer-events-none z-0 opacity-40"
                style={{
                    backgroundImage:
                        gridType === "polar"
                            ? "radial-gradient(circle at center, rgba(139,170,196,0.16) 0%, rgba(139,170,196,0.03) 50%, transparent 75%)"
                            : "linear-gradient(to right, rgba(59,130,246,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(59,130,246,0.05) 1px, transparent 1px)",
                    backgroundSize: gridType === "polar" ? "auto" : "30px 30px",
                }}
            />

            {/* Split Comparison View HUD Headers & Divider */}
            {visMode === "split_comparison" && (
                <>
                    {/* Center Divider Line */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-cyan-500/30 z-20 pointer-events-none transform -translate-x-1/2 flex items-center justify-center">
                        <div className="bg-[#050d1a] border border-cyan-500/50 rounded-full px-2 py-1 text-[9px] font-mono text-cyan-400 font-bold uppercase tracking-widest shadow-xl">
                            VS
                        </div>
                    </div>

                    {/* Left Pane HUD */}
                    <div className="absolute top-3 left-4 z-20 pointer-events-none flex items-center gap-2 bg-[#050d1a]/85 border border-cyan-500/40 rounded-lg px-3 py-1.5 backdrop-blur-md">
                        <Layers className="w-3.5 h-3.5 text-cyan-400" />
                        <div>
                            <p className="text-[10px] font-bold text-cyan-300 uppercase tracking-widest">
                                Height grid
                            </p>
                            <p className="text-[8px] text-slate-400 font-mono">
                                Height-grid representation ({normalizedCells.length} cells)
                            </p>
                        </div>
                    </div>

                    {/* Right Pane HUD */}
                    <div className="absolute top-3 right-4 z-20 pointer-events-none flex items-center gap-2 bg-[#050d1a]/85 border border-purple-500/40 rounded-lg px-3 py-1.5 backdrop-blur-md">
                        <Box className="w-3.5 h-3.5 text-purple-400" />
                        <div className="text-right">
                            <p className="text-[10px] font-bold text-purple-300 uppercase tracking-widest">
                                Sampled point cloud
                            </p>
                            <p className="text-[8px] text-slate-400 font-mono">
                                Sampled preview ({pointCloudData.length} points)
                            </p>
                        </div>
                    </div>
                </>
            )}

            {/* Single View Mode Active Badge */}
            {visMode !== "split_comparison" && (
                <div className="absolute top-3 left-4 z-20 pointer-events-none flex items-center gap-2 bg-[#050d1a]/80 border border-white/10 rounded-lg px-3 py-1.5 backdrop-blur-md">
                    {visMode === "grid_25d" ? (
                        <>
                            <Compass className="w-3.5 h-3.5 text-cyan-400" />
                            <span className="text-[10px] font-bold text-cyan-300 uppercase tracking-widest">
                                {gridType === "polar" ? "Polar height grid" : "Cartesian height grid"}
                            </span>
                        </>
                    ) : (
                        <>
                            <Box className="w-3.5 h-3.5 text-purple-400" />
                            <span className="text-[10px] font-bold text-purple-300 uppercase tracking-widest">
                                Sampled point cloud
                            </span>
                        </>
                    )}
                </div>
            )}

            {/* Deck.gl Canvas */}
            <DeckGL
                views={views}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                viewState={cameraViews}
                onViewStateChange={({viewId, viewState}) =>
                    setCameraViews(previous => ({...previous, [viewId]: viewState}))}
                controller={{
                    dragRotate: true,
                    dragPan: true,
                    scrollZoom: true,
                    doubleClickZoom: true,
                }}
                layers={layers}
                layerFilter={({layer, viewport}) => {
                    if (visMode !== 'split_comparison') return true;
                    const isPointPreview = layer.id.includes('3d');
                    return viewport.id === 'right-pane' ? isPointPreview : !isPointPreview;
                }}
                style={{ position: "absolute", inset: "0" }}
            />

            <div className="absolute right-4 top-16 z-30 flex gap-1 rounded-xl border border-white/10 bg-[#152130]/95 p-1 shadow-lg">
                {[['perspective', 'Perspective'], ['top', 'Top view']].map(([value, label]) =>
                    <button key={value} onClick={() => selectCamera(value)} aria-pressed={cameraPreset === value}
                        className={`rounded-lg px-3 py-2 text-[11px] transition-colors ${cameraPreset === value ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'}`}>{label}</button>)}
                <button onClick={() => selectCamera(cameraPreset)} className="rounded-lg px-3 py-2 text-[11px] text-slate-400 hover:text-white" aria-label="Reset camera">Reset</button>
            </div>
            <div className="pointer-events-none absolute bottom-20 left-4 z-20 text-[10px] leading-5 text-slate-400">
                Drag to orbit · Shift + drag to pan · Scroll to zoom<br/>
                Height scale 1:1 · Empty areas have no displayed samples
            </div>
            {/* Hover Tooltip */}
            {tooltip && (
                <div
                    className="absolute z-50 pointer-events-none transform -translate-y-full"
                    style={{ left: tooltip.x + 15, top: tooltip.y - 15 }}
                >
                    <div className="bg-[#0b1329]/95 border border-cyan-500/50 rounded-lg p-3 text-xs font-mono shadow-2xl backdrop-blur-md min-w-[230px]">
                        {tooltip.cell && (
                            <>
                                <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-white/10">
                                    <div className="flex items-center gap-2">
                                        <div
                                            className="w-2.5 h-2.5 rounded-sm"
                                            style={{
                                                backgroundColor: `rgba(${tooltip.cell.color?.slice(0, 3).join(",")}, 1)`,
                                            }}
                                        />
                                        <p className="text-cyan-400 font-bold uppercase tracking-wider">
                                            {tooltip.cell.label.replace(/_/g, " ")}
                                        </p>
                                    </div>
                                    {tooltip.cell.zone_id !== undefined && (
                                        <span className="text-[9px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 rounded px-1.5 py-0.5">
                                            ZONE {tooltip.cell.zone_id}
                                        </span>
                                    )}
                                </div>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-300">
                                    <span className="text-slate-500">Range (R):</span>
                                    <span className="font-semibold text-cyan-300">{tooltip.cell.r ?? 0} m</span>
                                    <span className="text-slate-500">Azimuth (θ):</span>
                                    <span className="font-semibold text-cyan-300">{tooltip.cell.azimuth_deg ?? 0}°</span>
                                    <span className="text-slate-500">Elevation (Z):</span>
                                    <span className="font-semibold text-white">{tooltip.cell.elev.toFixed(2)} m</span>
                                    <span className="text-slate-500">Height (ΔZ):</span>
                                    <span className="font-semibold text-white">{tooltip.cell.delta_z.toFixed(2)} m</span>
                                    {tooltip.cell.slope_deg !== undefined && (
                                        <>
                                            <span className="text-slate-500">Slope:</span>
                                            <span className={`font-semibold ${tooltip.cell.slope_deg > 15 ? "text-red-400" : "text-emerald-400"}`}>
                                                {tooltip.cell.slope_deg.toFixed(1)}°
                                            </span>
                                        </>
                                    )}
                                    {tooltip.cell.drivable !== undefined && (
                                        <>
                                            <span className="text-slate-500">Drivability:</span>
                                            <span className={`font-semibold ${tooltip.cell.drivable ? "text-emerald-400" : "text-red-400"}`}>
                                                {tooltip.cell.drivable ? "NAVIGABLE" : "BLOCKED"}
                                            </span>
                                        </>
                                    )}
                                    <span className="text-slate-500">Coords:</span>
                                    <span className="font-semibold text-slate-400">
                                        ({tooltip.cell.cx.toFixed(1)}, {tooltip.cell.cy.toFixed(1)})
                                    </span>
                                </div>
                            </>
                        )}

                        {tooltip.point && (
                            <>
                                <div className="flex items-center gap-2 mb-2 pb-1.5 border-b border-white/10">
                                    <div
                                        className="w-2.5 h-2.5 rounded-full"
                                        style={{
                                            backgroundColor: `rgba(${getColor(tooltip.point.label).slice(0, 3).join(",")}, 1)`,
                                        }}
                                    />
                                    <p className="text-purple-400 font-bold uppercase tracking-wider">
                                        Raw Point: {tooltip.point.label.replace(/_/g, " ")}
                                    </p>
                                </div>
                                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-300">
                                    <span className="text-slate-500">X Position:</span>
                                    <span className="font-semibold text-white">{tooltip.point.x.toFixed(2)} m</span>
                                    <span className="text-slate-500">Y Position:</span>
                                    <span className="font-semibold text-white">{tooltip.point.y.toFixed(2)} m</span>
                                    <span className="text-slate-500">Z Position:</span>
                                    <span className="font-semibold text-white">{tooltip.point.z.toFixed(2)} m</span>
                                    <span className="text-slate-500">Reflectivity:</span>
                                    <span className="font-semibold text-amber-300">
                                        {(tooltip.point.intensity * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
