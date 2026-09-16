// app/page.tsx
"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import {
    Shield,
    Radio,
    Crosshair,
    Layers,
    Compass,
    Grid,
    Box,
    Scale,
    Activity,
    Sliders,
    Zap,
    Gauge,
} from "lucide-react";
import TelemetryPanel from "./components/TelemetryPanel";
import ThreatTicker from "./components/ThreatTicker";
import ComparativeStudyModal from "./components/ComparativeStudyModal";
import FoveatedStratifiedModal from "./components/FoveatedStratifiedModal";
import { useLidarStream } from "./hooks/useLidarStream";
import type { GridType, VisualizationMode, CoordinateFrame } from "./components/DeckCanvas2D";

// Dynamically import DeckGL canvas — SSR incompatible (WebGL)
const DeckCanvas2D = dynamic(() => import("./components/DeckCanvas2D"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-[#030712]">
            <div className="flex flex-col items-center gap-4">
                <div className="relative w-16 h-16">
                    <div className="absolute inset-0 rounded-full border-2 border-cyan-500/30 animate-ping" />
                    <div
                        className="absolute inset-2 rounded-full border-2 border-cyan-400/60 animate-spin"
                        style={{ animationDuration: "1s" }}
                    />
                    <Crosshair className="absolute inset-0 m-auto w-6 h-6 text-cyan-400" />
                </div>
                <p className="text-cyan-500/60 text-xs font-mono uppercase tracking-widest">
                    Loading Perception Engine…
                </p>
            </div>
        </div>
    ),
});

export default function SentinelDashboard() {
    const {
        connected,
        playbackSpeed,
        setPlaybackSpeed,
        cells,
        polarCells,
        linearCells,
        pointCloud3D,
        comparativeStudy,
        stratifiedMetrics,
        egoVehicle,
        telemetry,
        threats,
    } = useLidarStream();

    // UI Configuration & Control State
    const [gridType, setGridType] = useState<GridType>("polar");
    const [visMode, setVisMode] = useState<VisualizationMode>("grid_25d");
    const [coordinateFrame, setCoordinateFrame] = useState<CoordinateFrame>("sensor");
    const [isStudyOpen, setIsStudyOpen] = useState(false);
    const [isFoveatedModalOpen, setIsFoveatedModalOpen] = useState(false);

    const activeDisplayCells = gridType === "linear" && linearCells.length > 0 ? linearCells : (polarCells.length > 0 ? polarCells : cells);

    return (
        <div className="h-screen w-screen bg-[#030712] text-white flex flex-col overflow-hidden font-mono">
            {/* ── Top Navigation Bar ─────────────────────────────────────────── */}
            <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 border-b border-white/5 bg-[#050d1a]/80 backdrop-blur-md z-20">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <Shield className="w-7 h-7 text-cyan-400" />
                        <div className="absolute inset-0 bg-cyan-400/20 blur-md rounded-full animate-pulse" />
                    </div>
                    <div>
                        <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-white">
                            SENTINEL<span className="text-cyan-400">-3D</span>
                        </h1>
                        <p className="text-[9px] text-slate-500 tracking-widest uppercase">
                            Defense Tactical Perception · DRDO
                        </p>
                    </div>
                </div>

                {/* Tactical Control Bar */}
                <div className="hidden lg:flex items-center gap-2 bg-[#08152b] p-1 rounded-xl border border-white/10">
                    {/* Playback Speed Multiplier (1x, 2x, 4x, 8x Turbo) */}
                    <div className="flex items-center bg-black/40 rounded-lg p-0.5 border border-white/5">
                        <span className="text-[9px] text-slate-500 px-1.5 flex items-center gap-1 font-bold">
                            <Zap className="w-2.5 h-2.5 text-amber-400" />
                            SPEED:
                        </span>
                        {[
                            { val: 1.0, label: "1x" },
                            { val: 2.0, label: "2x" },
                            { val: 4.0, label: "4x" },
                            { val: 8.0, label: "8x ⚡" },
                        ].map((s) => (
                            <button
                                key={s.val}
                                onClick={() => setPlaybackSpeed(s.val)}
                                className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                    playbackSpeed === s.val
                                        ? "bg-amber-500/25 text-amber-300 border border-amber-500/50 shadow-sm"
                                        : "text-slate-400 hover:text-white"
                                }`}
                            >
                                {s.label}
                            </button>
                        ))}
                    </div>

                    {/* Grid Type Selector (Polar vs Linear) */}
                    <div className="flex items-center bg-black/40 rounded-lg p-0.5 border border-white/5">
                        <button
                            onClick={() => setGridType("polar")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                gridType === "polar"
                                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            <Compass className="w-3 h-3" />
                            FOVEATED POLAR
                        </button>
                        <button
                            onClick={() => setGridType("linear")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                gridType === "linear"
                                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            <Grid className="w-3 h-3" />
                            LINEAR GRID
                        </button>
                    </div>

                    {/* Mode Selector (2.5D vs 3D vs Comparison) */}
                    <div className="flex items-center bg-black/40 rounded-lg p-0.5 border border-white/5">
                        <button
                            onClick={() => setVisMode("grid_25d")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                visMode === "grid_25d"
                                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            <Layers className="w-3 h-3" />
                            2.5D GRID
                        </button>
                        <button
                            onClick={() => setVisMode("pointcloud_3d")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                visMode === "pointcloud_3d"
                                    ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            <Box className="w-3 h-3" />
                            3D POINTS
                        </button>
                        <button
                            onClick={() => setVisMode("split_comparison")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                visMode === "split_comparison"
                                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            <Scale className="w-3 h-3" />
                            SPLIT VIEW
                        </button>
                    </div>

                    {/* Coordinate Frame (Sensor vs World Motion Compensation) */}
                    <div className="flex items-center bg-black/40 rounded-lg p-0.5 border border-white/5">
                        <button
                            onClick={() => setCoordinateFrame("sensor")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                coordinateFrame === "sensor"
                                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            SENSOR FRAME
                        </button>
                        <button
                            onClick={() => setCoordinateFrame("world")}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[10px] font-bold tracking-wider transition-colors ${
                                coordinateFrame === "world"
                                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                                    : "text-slate-400 hover:text-white"
                            }`}
                        >
                            WORLD (TRACKED)
                        </button>
                    </div>

                    {/* Foveated Stratified Metrics Modal Trigger */}
                    <button
                        onClick={() => setIsFoveatedModalOpen(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider bg-gradient-to-r from-purple-500/20 to-indigo-500/20 hover:from-purple-500/30 hover:to-indigo-500/30 text-purple-300 border border-purple-500/40 transition-all shadow-lg"
                    >
                        <Compass className="w-3.5 h-3.5 text-purple-400" />
                        Foveated Zones & Radio
                    </button>

                    {/* 3D vs 2.5D Study Modal Trigger */}
                    <button
                        onClick={() => setIsStudyOpen(true)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider bg-gradient-to-r from-cyan-500/20 to-blue-500/20 hover:from-cyan-500/30 hover:to-blue-500/30 text-cyan-300 border border-cyan-500/40 transition-all shadow-lg"
                    >
                        <Scale className="w-3.5 h-3.5 text-cyan-400" />
                        3D vs 2.5D Study
                    </button>
                </div>

                {/* Status indicators */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="relative flex h-2 w-2">
                            <span
                                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                                    connected ? "bg-emerald-400" : "bg-red-400"
                                }`}
                            />
                            <span
                                className={`relative inline-flex rounded-full h-2 w-2 ${
                                    connected ? "bg-emerald-500" : "bg-red-500"
                                }`}
                            />
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">
                            {connected ? "STREAM ACTIVE" : "DISCONNECTED"}
                        </span>
                    </div>

                    <div className="hidden sm:flex items-center gap-1 bg-cyan-950/40 border border-cyan-500/30 rounded px-2 py-0.5 text-[9px] text-cyan-300 font-mono">
                        <Radio className="w-2.5 h-2.5 text-cyan-400 animate-pulse" />
                        <span>TACTICAL LINK ACTIVE</span>
                    </div>
                </div>
            </header>

            {/* ── Main Content ───────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden">
                {/* ── Left: Deck.gl Canvas (70%) ──────────────────────────────── */}
                <div className="relative flex-[7] min-w-0 overflow-hidden border-r border-white/5">
                    {/* Corner brackets */}
                    {(["top-2 left-2", "top-2 right-2", "bottom-2 left-2", "bottom-2 right-2"] as const).map(
                        (pos) => (
                            <div
                                key={pos}
                                className={`absolute ${pos} w-5 h-5 border-cyan-500/40 pointer-events-none z-10`}
                                style={{
                                    borderTopWidth: pos.startsWith("top") ? 1 : 0,
                                    borderBottomWidth: pos.startsWith("bottom") ? 1 : 0,
                                    borderLeftWidth: pos.endsWith("left-2") ? 1 : 0,
                                    borderRightWidth: pos.endsWith("right-2") ? 1 : 0,
                                }}
                            />
                        )
                    )}

                    {/* Canvas label */}
                    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
                        <span className="text-[9px] font-mono uppercase tracking-[0.35em] text-slate-400 bg-[#030712]/90 px-3 py-1 rounded-full border border-white/10 shadow-lg">
                            {gridType === "polar" ? "Foveated Multi-Resolution Polar Grid" : "Linear Cartesian Grid"} · {visMode === "grid_25d" ? "2.5D Volumetric" : visMode === "pointcloud_3d" ? "3D Point Cloud" : "3D/2.5D Split Comparison"}
                        </span>
                    </div>

                    {/* Frame & Compression metrics */}
                    {telemetry && (
                        <div className="absolute top-3 right-4 z-10 text-[9px] font-mono text-slate-400 bg-[#030712]/80 px-2.5 py-1 rounded border border-white/5 flex items-center gap-2 pointer-events-none">
                            <span>{activeDisplayCells.length.toLocaleString()} cells</span>
                            <span>·</span>
                            <span className="text-cyan-400">{telemetry.latencyMs.toFixed(1)} ms</span>
                            <span>·</span>
                            <span className="text-emerald-400">{telemetry.memorySavedPct.toFixed(1)}% Saved</span>
                        </div>
                    )}

                    <DeckCanvas2D
                        cells={polarCells.length > 0 ? polarCells : cells}
                        linearCells={linearCells}
                        pointCloud3D={pointCloud3D}
                        threats={threats}
                        egoVehicle={egoVehicle}
                        gridType={gridType}
                        visMode={visMode}
                        coordinateFrame={coordinateFrame}
                    />
                </div>

                {/* ── Right: Sidebar (30%) ────────────────────────────────────── */}
                <div className="flex-[3] min-w-0 flex flex-col gap-4 overflow-hidden p-4 bg-[#050d1a]/60">
                    {/* Telemetry Panel */}
                    <div className="flex-[4] min-h-0 overflow-hidden">
                        <TelemetryPanel metrics={telemetry} />
                    </div>

                    {/* Divider */}
                    <div className="flex-shrink-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                    {/* Threat Ticker */}
                    <div className="flex-[5] min-h-0 overflow-hidden">
                        <ThreatTicker
                            cells={activeDisplayCells}
                            backendThreats={threats}
                            maxEvents={50}
                        />
                    </div>
                </div>
            </div>

            {/* ── Footer ──────────────────────────────────────────────────────── */}
            <footer className="flex-shrink-0 flex items-center justify-between px-6 py-1.5 border-t border-white/5 bg-[#050d1a]/80 text-[9px] text-slate-600">
                <span className="uppercase tracking-widest">
                    SENTINEL-3D v2.0 · DRDO · Tactical Motion-Compensated Foveated LiDAR Perception
                </span>
                <div className="flex items-center gap-4">
                    <span>Frame: {egoVehicle?.frame_id ?? 0}</span>
                    <span>Speed: {egoVehicle?.speed_kmh.toFixed(1) ?? "36.0"} km/h ({playbackSpeed}x)</span>
                    <span>ws://127.0.0.1:8005/ws/stream_map</span>
                </div>
            </footer>

            {/* ── Foveated Stratified & Tactical Radio Modal ─────────────────── */}
            <FoveatedStratifiedModal
                isOpen={isFoveatedModalOpen}
                onClose={() => setIsFoveatedModalOpen(false)}
                data={stratifiedMetrics}
            />

            {/* ── 3D vs 2.5D Comparative Study Modal ───────────────────────── */}
            <ComparativeStudyModal
                isOpen={isStudyOpen}
                onClose={() => setIsStudyOpen(false)}
                data={comparativeStudy}
            />
        </div>
    );
}
