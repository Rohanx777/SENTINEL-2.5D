// app/components/FoveatedStratifiedModal.tsx
"use client";

import React from "react";
import {
    X,
    Layers,
    Radio,
    ShieldAlert,
    TrendingDown,
    Activity,
    Compass,
    CheckCircle2,
    Sliders,
    Zap,
    Cpu,
} from "lucide-react";
import type { StratifiedMetricsData } from "../hooks/useLidarStream";

interface Props {
    isOpen: boolean;
    onClose: () => void;
    data: StratifiedMetricsData | null;
}

export default function FoveatedStratifiedModal({ isOpen, onClose, data }: Props) {
    if (!isOpen) return null;

    const zones = data?.zones ?? [
        {
            zone_id: 0,
            name: "Safety Core (Near)",
            range_meters: "0.5m - 10.0m",
            resolution: "0.5m radial × 2.8° azimuth (5-10cm tangential)",
            accuracy: "±2 cm (Micro-Terrain Safety)",
            points_ingested: 50000,
            cells_generated: 450,
            compression_ratio_pct: 99.1,
            threats_detected: 2,
            purpose: "Potholes, curbs, pedestrians, negative obstacles, micro-terrain",
        },
        {
            zone_id: 1,
            name: "Tactical Corridor (Mid)",
            range_meters: "10.0m - 30.0m",
            resolution: "1.0m radial × 5.6° azimuth (~25-40cm tangential)",
            accuracy: "±8 cm (Maneuver Planning)",
            points_ingested: 45000,
            cells_generated: 280,
            compression_ratio_pct: 99.4,
            threats_detected: 1,
            purpose: "Vehicle maneuver planning, obstacle avoidance, convoy tracking",
        },
        {
            zone_id: 2,
            name: "Perimeter Surveillance (Far)",
            range_meters: "30.0m - 100.0m",
            resolution: "2.5m radial × 11.25° azimuth (~1-2m tangential)",
            accuracy: "±20 cm (Perimeter Awareness)",
            points_ingested: 25000,
            cells_generated: 120,
            compression_ratio_pct: 99.5,
            threats_detected: 0,
            purpose: "Perimeter buildings, terrain elevation, long-range warning",
        },
    ];

    const radio = data?.tactical_radio_simulation ?? {
        cbr_64kbps: {
            link_name: "Combat Net Radio (CNR - 64 kbps)",
            tx_time_3d_sec: 255.0,
            tx_time_foveated_ms: 780.0,
            status: "OPERATIONAL (Sub-second)",
        },
        sdr_256kbps: {
            link_name: "Tactical SDR Mesh (256 kbps)",
            tx_time_3d_sec: 63.8,
            tx_time_foveated_ms: 195.0,
            status: "REAL-TIME (< 250ms)",
        },
        link16_1200kbps: {
            link_name: "Link-16 / High-Speed Tactical Data Link (1.2 Mbps)",
            tx_time_3d_sec: 13.6,
            tx_time_foveated_ms: 41.6,
            status: "STREAMING (30 FPS)",
        },
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-8 bg-black/80 backdrop-blur-md animate-fadeIn">
            <div className="relative w-full max-w-5xl max-h-[90vh] bg-[#050d1a] border border-cyan-500/40 rounded-2xl shadow-2xl flex flex-col overflow-hidden font-mono text-white">
                {/* ── Modal Header ────────────────────────────────────────────── */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#071326]/90">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                            <Compass className="w-5 h-5" />
                        </div>
                        <div>
                            <h2 className="text-sm font-bold tracking-[0.25em] uppercase text-white flex items-center gap-2">
                                Adaptive Foveated Grid & Tactical Link Simulator
                            </h2>
                            <p className="text-[10px] text-slate-400">
                                DRDO Distance-Stratified Precision & Military Tactical Radio Benchmark
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* ── Modal Body (Scrollable) ─────────────────────────────────── */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Zone Architecture Overview */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {zones.map((z) => (
                            <div
                                key={z.zone_id}
                                className={`p-4 rounded-xl border flex flex-col gap-2 ${
                                    z.zone_id === 0
                                        ? "border-cyan-500/40 bg-cyan-950/20"
                                        : z.zone_id === 1
                                        ? "border-blue-500/40 bg-blue-950/20"
                                        : "border-purple-500/40 bg-purple-950/20"
                                }`}
                            >
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold uppercase tracking-wider text-white">
                                        {z.name}
                                    </span>
                                    <span
                                        className={`text-[9px] px-2 py-0.5 rounded font-bold uppercase ${
                                            z.zone_id === 0
                                                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                                                : z.zone_id === 1
                                                ? "bg-blue-500/20 text-blue-300 border border-blue-500/40"
                                                : "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                                        }`}
                                    >
                                        {z.range_meters}
                                    </span>
                                </div>
                                <div className="space-y-1 text-[11px] text-slate-300">
                                    <p>
                                        <span className="text-slate-500">Resolution:</span> {z.resolution}
                                    </p>
                                    <p>
                                        <span className="text-slate-500">Accuracy:</span>{" "}
                                        <strong className="text-emerald-300">{z.accuracy}</strong>
                                    </p>
                                    <p>
                                        <span className="text-slate-500">Ingested:</span>{" "}
                                        {z.points_ingested.toLocaleString()} pts →{" "}
                                        <strong className="text-cyan-300">{z.cells_generated} cells</strong>
                                    </p>
                                    <p>
                                        <span className="text-slate-500">Compression:</span>{" "}
                                        <span className="text-emerald-400 font-bold">
                                            {z.compression_ratio_pct}%
                                        </span>
                                    </p>
                                </div>
                                <div className="mt-auto pt-2 border-t border-white/5 text-[10px] text-slate-400 italic">
                                    Target: {z.purpose}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Tactical Radio Link Simulation Table */}
                    <div className="rounded-xl border border-white/10 bg-[#08152b]/60 overflow-hidden">
                        <div className="px-4 py-3 bg-[#0a1a36] border-b border-white/10 flex items-center justify-between">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-cyan-400 flex items-center gap-2">
                                <Radio className="w-4 h-4" /> Tactical Radio Link Throughput Simulation
                            </h3>
                            <span className="text-[10px] text-slate-400">
                                Over-the-Air Military Datalink Feasibility
                            </span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-xs text-left">
                                <thead className="border-b border-white/10 text-[10px] uppercase text-slate-400 bg-white/[0.02]">
                                    <tr>
                                        <th className="px-4 py-3 font-semibold">Military Radio Standard</th>
                                        <th className="px-4 py-3 font-semibold text-rose-400">Raw 3D Transmission Time</th>
                                        <th className="px-4 py-3 font-semibold text-cyan-300">Foveated 2.5D Transmission Time</th>
                                        <th className="px-4 py-3 font-semibold text-emerald-400">Tactical Status</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    <tr className="hover:bg-white/[0.02]">
                                        <td className="px-4 py-3 font-medium text-slate-200">{radio.cbr_64kbps.link_name}</td>
                                        <td className="px-4 py-3 text-rose-400 font-semibold">{radio.cbr_64kbps.tx_time_3d_sec.toFixed(1)} s (Unusable)</td>
                                        <td className="px-4 py-3 text-cyan-300 font-bold">{radio.cbr_64kbps.tx_time_foveated_ms.toFixed(1)} ms</td>
                                        <td className="px-4 py-3 text-emerald-400 font-semibold">{radio.cbr_64kbps.status}</td>
                                    </tr>
                                    <tr className="hover:bg-white/[0.02]">
                                        <td className="px-4 py-3 font-medium text-slate-200">{radio.sdr_256kbps.link_name}</td>
                                        <td className="px-4 py-3 text-rose-400 font-semibold">{radio.sdr_256kbps.tx_time_3d_sec.toFixed(1)} s (Severe Lag)</td>
                                        <td className="px-4 py-3 text-cyan-300 font-bold">{radio.sdr_256kbps.tx_time_foveated_ms.toFixed(1)} ms</td>
                                        <td className="px-4 py-3 text-emerald-400 font-semibold">{radio.sdr_256kbps.status}</td>
                                    </tr>
                                    <tr className="hover:bg-white/[0.02]">
                                        <td className="px-4 py-3 font-medium text-slate-200">{radio.link16_1200kbps.link_name}</td>
                                        <td className="px-4 py-3 text-amber-400 font-semibold">{radio.link16_1200kbps.tx_time_3d_sec.toFixed(1)} s</td>
                                        <td className="px-4 py-3 text-cyan-300 font-bold">{radio.link16_1200kbps.tx_time_foveated_ms.toFixed(1)} ms</td>
                                        <td className="px-4 py-3 text-emerald-400 font-semibold">{radio.link16_1200kbps.status}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Technical Innovations Breakdown */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 rounded-xl border border-white/5 bg-white/[0.02] flex flex-col gap-2">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                                <CheckCircle2 className="w-4 h-4 text-cyan-400" />
                                2:1 Hierarchical Boundary Snapping
                            </h4>
                            <p className="text-[11px] text-slate-300 leading-relaxed">
                                Avoids topological tearing and overlapping artifacts at zone transitions by enforcing exact <strong>2:1 integer sector subdivisions</strong> (128 &rarr; 64 &rarr; 32 sectors). Each sector in Zone 1 exactly splits into 2 sub-sectors in Zone 0, ensuring zero data loss during 3D &rarr; 2.5D projection.
                            </p>
                        </div>

                        <div className="p-4 rounded-xl border border-white/5 bg-white/[0.02] flex flex-col gap-2">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-purple-400 flex items-center gap-1.5">
                                <ShieldAlert className="w-4 h-4 text-purple-400" />
                                Negative Obstacle & Micro-Terrain Analyzer
                            </h4>
                            <p className="text-[11px] text-slate-300 leading-relaxed">
                                Ingests high-density Safety Core data to detect ground depressions (&Delta;Z &lt; -0.22m) identifying potholes, ditches, and craters, while calculating terrain slope gradients (&theta; &le; 15&deg;) for autonomous drivability.
                            </p>
                        </div>
                    </div>
                </div>

                {/* ── Modal Footer ────────────────────────────────────────────── */}
                <div className="px-6 py-3 border-t border-white/10 bg-[#071326] flex items-center justify-between">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest">
                        SENTINEL-3D · Foveated Perception Engine
                    </span>
                    <button
                        onClick={onClose}
                        className="px-4 py-1.5 rounded-lg text-xs font-bold uppercase tracking-widest bg-cyan-500 hover:bg-cyan-400 text-black transition-colors"
                    >
                        Dismiss Analysis
                    </button>
                </div>
            </div>
        </div>
    );
}
