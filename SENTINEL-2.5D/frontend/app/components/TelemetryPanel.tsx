// app/components/TelemetryPanel.tsx
"use client";

import React from "react";
import { Cpu, Gauge, Timer, MemoryStick } from "lucide-react";
import type { TelemetryMetrics } from "../hooks/useLidarStream";

interface Props {
    metrics: TelemetryMetrics | null;
}

function MetricCard({
    icon,
    label,
    value,
    sub,
    accent,
}: {
    icon: React.ReactNode;
    label: string;
    value: string;
    sub?: string;
    accent?: string;
}) {
    return (
        <div className="relative overflow-hidden rounded-xl border border-white/5 bg-white/[0.03] p-4 flex flex-col gap-2 group hover:border-cyan-500/30 transition-colors duration-300">
            {/* Glow blob */}
            <div className="absolute -top-6 -right-6 w-20 h-20 rounded-full bg-cyan-500/10 blur-2xl group-hover:bg-cyan-500/20 transition-all duration-500 pointer-events-none" />
            <div className="flex items-center gap-2 text-slate-400 text-xs font-mono uppercase tracking-widest">
                <span className="text-cyan-500/80">{icon}</span>
                {label}
            </div>
            <div className={`text-2xl font-bold font-mono ${accent ?? "text-white"}`}>
                {value}
            </div>
            {sub && (
                <div className="text-[10px] text-slate-500 font-mono">{sub}</div>
            )}
        </div>
    );
}

const DEFAULT: TelemetryMetrics = {
    engine: "CPU OPENMP FALLBACK",
    fps: 0,
    latencyMs: 0,
    memorySavedPct: 0,
};

export default function TelemetryPanel({ metrics }: Props) {
    const data = metrics ?? DEFAULT;

    const isCuda = data.engine === "CUDA GPU TIER 1";

    return (
        <div className="flex flex-col gap-3 h-full">
            {/* Header */}
            <div className="flex items-center justify-between px-1">
                <h2 className="text-xs font-mono uppercase tracking-[0.25em] text-slate-400">
                    Telemetry
                </h2>
                <span className="text-[9px] font-mono text-slate-600 uppercase">
                    LIVE
                </span>
            </div>

            {/* Engine status badge */}
            <div
                className={`relative flex items-center gap-3 rounded-xl border px-4 py-3 overflow-hidden ${isCuda
                        ? "border-cyan-500/40 bg-cyan-950/20"
                        : "border-amber-500/40 bg-amber-950/20"
                    }`}
            >
                <div
                    className={`absolute inset-0 opacity-10 ${isCuda
                            ? "bg-gradient-to-r from-cyan-600 to-blue-700"
                            : "bg-gradient-to-r from-amber-600 to-orange-700"
                        }`}
                />
                <Cpu
                    className={`w-5 h-5 relative z-10 ${isCuda ? "text-cyan-400" : "text-amber-400"
                        }`}
                />
                <div className="relative z-10">
                    <p
                        className={`text-xs font-mono font-bold tracking-widest ${isCuda ? "text-cyan-300" : "text-amber-300"
                            }`}
                    >
                        {data.engine}
                    </p>
                    <p className="text-[10px] text-slate-500 font-mono">
                        {isCuda
                            ? "Hardware-accelerated inference"
                            : "Software parallel fallback"}
                    </p>
                </div>
                {/* Pulsing dot */}
                <div className="relative z-10 ml-auto">
                    <span
                        className={`relative flex h-2.5 w-2.5`}
                    >
                        <span
                            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isCuda ? "bg-cyan-400" : "bg-amber-400"
                                }`}
                        />
                        <span
                            className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isCuda ? "bg-cyan-400" : "bg-amber-400"
                                }`}
                        />
                    </span>
                </div>
            </div>

            {/* Metric cards grid */}
            <div className="grid grid-cols-2 gap-2 flex-1">
                <MetricCard
                    icon={<Gauge className="w-3.5 h-3.5" />}
                    label="FPS"
                    value={data.fps.toFixed(0)}
                    sub="frames / second"
                    accent={data.fps >= 25 ? "text-green-400" : data.fps >= 15 ? "text-amber-400" : "text-red-400"}
                />
                <MetricCard
                    icon={<Timer className="w-3.5 h-3.5" />}
                    label="Latency"
                    value={`${data.latencyMs.toFixed(1)}`}
                    sub="milliseconds"
                    accent={data.latencyMs < 50 ? "text-green-400" : data.latencyMs < 100 ? "text-amber-400" : "text-red-400"}
                />
            </div>

            {/* Memory reduction progress */}
            <div className="rounded-xl border border-white/5 bg-white/[0.03] p-4">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-slate-400">
                        <MemoryStick className="w-3.5 h-3.5 text-cyan-500/80" />
                        Memory Saved
                    </div>
                    <span className="text-sm font-mono font-bold text-cyan-300">
                        {data.memorySavedPct.toFixed(1)}%
                    </span>
                </div>
                <div className="relative h-2 w-full rounded-full bg-white/5 overflow-hidden">
                    <div
                        className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500 ease-out"
                        style={{ width: `${Math.min(100, data.memorySavedPct)}%` }}
                    />
                    {/* Shimmer */}
                    <div
                        className="absolute top-0 h-full w-12 bg-gradient-to-r from-transparent via-white/30 to-transparent rounded-full"
                        style={{
                            left: `calc(${Math.min(100, data.memorySavedPct)}% - 24px)`,
                            transition: "left 0.5s ease-out",
                        }}
                    />
                </div>
                <p className="text-[10px] text-slate-600 font-mono mt-1.5">
                    vs. full-resolution baseline
                </p>
            </div>
        </div>
    );
}
