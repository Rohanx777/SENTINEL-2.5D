// app/components/ThreatTicker.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, ShieldAlert, ChevronRight, Zap, Navigation, Crosshair, Clock } from "lucide-react";
import type { CellTuple, ThreatObject } from "../hooks/useLidarStream";

// ─── Internal event shape ────────────────────────────────────────────────────
interface ThreatEvent {
    instanceId: string;
    threatId: string;
    id: string;
    timestamp: string;
    type: "DROP_OFF" | "CLOSE_RANGE" | "NEGATIVE_OBSTACLE" | "DYNAMIC_KINEMATIC" | "BACKEND";
    label: string;
    severity: "MODERATE" | "HIGH" | "CRITICAL";
    detail: string;
    ttc?: number;
    relSpeed?: number;
    heading?: number;
    source: "backend" | "client";
}

interface Props {
    cells: CellTuple[];
    backendThreats?: ThreatObject[];
    maxEvents?: number;
}

let eventCounter = 0;

function buildTimestamp(): string {
    return new Date().toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 2,
    } as Intl.DateTimeFormatOptions);
}

function makeId(): string {
    return (++eventCounter).toString(16).padStart(4, "0").toUpperCase();
}

// ── Convert backend ThreatObject → ThreatEvent ──────────────────────────────
function adaptBackendThreats(threats: ThreatObject[]): ThreatEvent[] {
    const ts = buildTimestamp();
    return threats.map((t, idx) => {
        const depth = t.depth_m ?? 0;
        const ttc = t.time_to_collision_sec;
        const isCritical = (ttc !== undefined && ttc < 3.0) || t.distance_m < 6.0 || depth > 0.4 || t.threat_level === "CRITICAL";
        const isHigh = (ttc !== undefined && ttc < 7.0) || t.distance_m < 18.0 || depth > 0.2 || t.threat_level === "HIGH";

        const [bx, by] = t.coordinates || [0, 0];
        let detailStr = `dist=${t.distance_m.toFixed(1)}m @ (${bx.toFixed(1)},${by.toFixed(1)})`;
        if (t.time_to_collision_sec !== undefined && t.time_to_collision_sec < 50.0) {
            detailStr += ` | TTC=${t.time_to_collision_sec.toFixed(1)}s | v_rel=${t.relative_speed_kmh ?? 0} km/h`;
        }
        if (t.type === "NEGATIVE_OBSTACLE" && depth > 0) {
            detailStr += ` | depth=${depth.toFixed(2)}m`;
        }

        const baseId = t.id ?? `THREAT-${(idx + 1).toString().padStart(2, "0")}`;
        const uniqueInstanceId = `${baseId}-${makeId()}-${Date.now()}`;

        return {
            instanceId: uniqueInstanceId,
            threatId: baseId,
            id: baseId,
            timestamp: ts,
            type: t.type === "NEGATIVE_OBSTACLE" ? "NEGATIVE_OBSTACLE" : (t.velocity_mps ? "DYNAMIC_KINEMATIC" : "BACKEND"),
            label: t.type.replace(/_/g, " "),
            severity: isCritical ? "CRITICAL" : (isHigh ? "HIGH" : "MODERATE"),
            detail: detailStr,
            ttc: t.time_to_collision_sec,
            relSpeed: t.relative_speed_kmh,
            heading: t.heading_deg,
            source: "backend",
        };
    });
}

// ── Client-side detection from grid cells (fallback / supplemental) ──────────
function detectFromCells(cells: CellTuple[]): ThreatEvent[] {
    const events: ThreatEvent[] = [];
    const ts = buildTimestamp();

    const sample = cells.length > 500
        ? cells.filter((_, i) => i % Math.ceil(cells.length / 500) === 0)
        : cells;

    for (const cell of sample) {
        let x: number, y: number, deltaZ: number, label: string, dist: number, slopeDeg: number;

        if (Array.isArray(cell)) {
            [x, y, , deltaZ, label] = cell as unknown as [number, number, number, number, string, number];
            dist = Math.sqrt(x * x + y * y);
            slopeDeg = 0;
        } else {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const c = cell as any;
            x = c.cx ?? 0; y = c.cy ?? 0;
            deltaZ = c.delta_z ?? 0;
            label = c.label ?? "unknown";
            slopeDeg = c.slope_deg ?? 0;
            dist = Math.sqrt(x * x + y * y);
        }

        if (label === "negative_obstacle") {
            const uid = `POTHOLE-${makeId()}`;
            events.push({
                instanceId: `${uid}-${Date.now()}`,
                threatId: uid,
                id: uid,
                timestamp: ts,
                type: "NEGATIVE_OBSTACLE",
                label: "POTHOLE / DITCH HAZARD",
                severity: deltaZ > 0.35 ? "CRITICAL" : "HIGH",
                detail: `depth=${deltaZ.toFixed(2)}m @ (${x.toFixed(1)},${y.toFixed(1)})`,
                source: "client",
            });
        } else if (deltaZ > 1.2 || slopeDeg > 20) {
            const uid = `DROPOFF-${makeId()}`;
            events.push({
                instanceId: `${uid}-${Date.now()}`,
                threatId: uid,
                id: uid,
                timestamp: ts,
                type: "DROP_OFF",
                label: String(label),
                severity: deltaZ > 2.0 ? "CRITICAL" : "HIGH",
                detail: `ΔZ=+${deltaZ.toFixed(2)}m (slope=${slopeDeg}°) @ (${x.toFixed(1)},${y.toFixed(1)})`,
                source: "client",
            });
        }
    }

    return events;
}

// ── Label for event type ─────────────────────────────────────────────────────
function typeLabel(type: ThreatEvent["type"]): string {
    switch (type) {
        case "DROP_OFF": return "STEEP OBSTACLE / DROP-OFF";
        case "CLOSE_RANGE": return "CLOSE PROXIMITY HAZARD";
        case "NEGATIVE_OBSTACLE": return "NEGATIVE OBSTACLE / POTHOLE";
        case "DYNAMIC_KINEMATIC": return "INTERCEPTING DYNAMIC TARGET";
        default: return "TACTICAL THREAT DETECTED";
    }
}

// ── Component ────────────────────────────────────────────────────────────────
export default function ThreatTicker({
    cells,
    backendThreats,
    maxEvents = 60,
}: Props) {
    const [events, setEvents] = useState<ThreatEvent[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Sync live threats into feed
    useEffect(() => {
        if (backendThreats && backendThreats.length > 0) {
            const currentThreats = adaptBackendThreats(backendThreats);
            setEvents(currentThreats.slice(0, maxEvents));
        } else if (cells && cells.length > 0) {
            const clientDetected = detectFromCells(cells);
            if (clientDetected.length > 0) {
                setEvents(clientDetected.slice(0, maxEvents));
            }
        }
    }, [backendThreats, cells, maxEvents]);

    // Auto-scroll to top on new event
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = 0;
    }, [events.length]);

    const critCount = events.filter((e) => e.severity === "CRITICAL").length;

    return (
        <div className="flex flex-col h-full">
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-1 mb-2">
                <div className="flex items-center gap-2">
                    <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                    <h2 className="text-xs font-mono uppercase tracking-[0.25em] text-slate-400">
                        Threat Kinematics Feed
                    </h2>
                    {critCount > 0 && (
                        <span className="flex items-center gap-0.5 text-[8px] font-mono font-bold text-red-400 animate-pulse bg-red-950/40 px-1.5 py-0.5 rounded border border-red-500/30">
                            <Zap className="w-2.5 h-2.5" />
                            {critCount} CRIT ALERT
                        </span>
                    )}
                </div>
                <span className="text-[9px] font-mono bg-red-500/20 text-red-400 border border-red-500/30 rounded px-1.5 py-0.5">
                    {events.length} ACTIVE
                </span>
            </div>

            {/* ── Scrolling log ── */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto space-y-1.5 pr-1"
                style={{
                    scrollbarWidth: "thin",
                    scrollbarColor: "rgba(255,255,255,0.08) transparent",
                }}
            >
                {events.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
                        <Crosshair className="w-8 h-8 opacity-20" />
                        <p className="text-xs font-mono">Perimeter Clear · Monitoring</p>
                        <p className="text-[9px] text-slate-700 font-mono">
                            Foveated zones scanning for micro-hazards & dynamic targets…
                        </p>
                    </div>
                ) : (
                    events.map((ev, idx) => (
                        <div
                            key={ev.instanceId || `${ev.threatId}-${idx}-${ev.timestamp}`}
                            className={`relative overflow-hidden flex items-start gap-2.5 rounded-lg border px-3 py-2 text-xs font-mono ${ev.severity === "CRITICAL"
                                ? "border-red-500/50 bg-red-950/30 ring-1 ring-inset ring-red-500/20"
                                : ev.severity === "HIGH"
                                    ? "border-amber-500/30 bg-amber-950/20"
                                    : "border-cyan-500/20 bg-cyan-950/10"
                                } ${idx === 0 ? "animate-pulse" : ""}`}
                        >
                            {/* Left severity bar */}
                            <div
                                className={`absolute left-0 top-0 w-1 h-full ${ev.severity === "CRITICAL"
                                    ? "bg-red-500"
                                    : ev.severity === "HIGH"
                                        ? "bg-amber-400"
                                        : "bg-cyan-400"
                                    }`}
                            />

                            <AlertTriangle
                                className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${ev.severity === "CRITICAL"
                                    ? "text-red-400"
                                    : ev.severity === "HIGH"
                                        ? "text-amber-400"
                                        : "text-cyan-400"
                                    }`}
                            />

                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-2 flex-wrap">
                                    <span
                                        className={`font-bold uppercase tracking-wider text-[10px] ${ev.severity === "CRITICAL"
                                            ? "text-red-300"
                                            : ev.severity === "HIGH"
                                                ? "text-amber-300"
                                                : "text-cyan-300"
                                            }`}
                                    >
                                        {typeLabel(ev.type)}
                                    </span>
                                    <span className="text-slate-500 text-[9px] flex-shrink-0 tabular-nums">
                                        {ev.timestamp}
                                    </span>
                                </div>

                                <div className="flex items-center gap-1 text-slate-400 mt-0.5">
                                    <ChevronRight className="w-2.5 h-2.5 flex-shrink-0 text-slate-600" />
                                    <span className="text-white font-medium truncate">{ev.label}</span>
                                    <span className="text-slate-600">·</span>
                                    <span className="text-slate-400 truncate">{ev.detail}</span>
                                </div>

                                {/* TTC & Kinematics Indicators */}
                                <div className="flex items-center gap-2 mt-1">
                                    <span
                                        className={`text-[8px] px-1.5 py-0.5 rounded uppercase font-bold tracking-widest ${ev.severity === "CRITICAL"
                                            ? "bg-red-500/20 text-red-400 border border-red-500/40"
                                            : ev.severity === "HIGH"
                                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                                : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                                            }`}
                                    >
                                        {ev.severity}
                                    </span>

                                    {ev.ttc !== undefined && ev.ttc < 50 && (
                                        <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono font-bold flex items-center gap-1 ${ev.ttc < 3.0 ? "bg-red-950 text-red-300 border border-red-500" : "bg-black/40 text-amber-300 border border-white/10"}`}>
                                            <Clock className="w-2.5 h-2.5" />
                                            TTC: {ev.ttc.toFixed(1)}s
                                        </span>
                                    )}

                                    {ev.heading !== undefined && (
                                        <span className="text-[8px] px-1.5 py-0.5 rounded font-mono text-slate-400 bg-black/40 border border-white/10 flex items-center gap-0.5">
                                            <Navigation className="w-2.5 h-2.5" />
                                            {ev.heading}°
                                        </span>
                                    )}

                                    <span className="text-[8px] text-slate-600 font-mono ml-auto">
                                        ID:{ev.threatId}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
