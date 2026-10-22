"use client";
import AnalysisDialog from "./AnalysisDialog";
import type { StratifiedMetricsData } from "../hooks/useLidarStream";

export default function FoveatedStratifiedModal({isOpen,onClose,data}:{isOpen:boolean;onClose:()=>void;data:StratifiedMetricsData|null}) {
    const radio=data?.tactical_radio_simulation;
    const links=radio?[["64 kbps",radio.cbr_64kbps],["256 kbps",radio.sdr_256kbps],["1.2 Mbps",radio.link16_1200kbps]] as const:[];
    return <AnalysisDialog isOpen={isOpen} onClose={onClose} title="Grid resolution and link estimates" subtitle="Adaptive spatial aggregation and illustrative payload transfer times.">
        {!data?<p className="text-sm text-slate-400">Waiting for stream data.</p>:<>
            <div className="grid gap-3 md:grid-cols-3">{data.zones.map(zone=><div key={zone.zone_id} className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                <h3 className="font-semibold text-cyan-200">{zone.name}</h3><p className="mt-1 text-xs text-slate-400">{zone.range_meters}</p>
                <dl className="mt-4 space-y-2 text-xs"><div><dt className="text-slate-400">Resolution</dt><dd>{zone.resolution}</dd></div><div><dt className="text-slate-400">Source points / output cells</dt><dd>{zone.points_ingested.toLocaleString()} / {zone.cells_generated.toLocaleString()}</dd></div><div><dt className="text-slate-400">Accuracy</dt><dd>Not evaluated</dd></div></dl>
            </div>)}</div>
            <div className="overflow-x-auto rounded-xl border border-white/10"><table className="w-full text-left text-sm"><thead className="bg-white/5 text-slate-400"><tr><th className="p-3">Assumed link rate</th><th className="p-3">Raw scan</th><th className="p-3">JSON grid</th></tr></thead><tbody className="divide-y divide-white/10">{links.map(([name,link])=><tr key={name}><td className="p-3">{name}</td><td className="p-3">{(link.tx_time_3d_sec*1000).toFixed(1)} ms</td><td className="p-3">{link.tx_time_foveated_ms.toFixed(1)} ms</td></tr>)}</tbody></table></div>
            <p className="text-xs leading-relaxed text-slate-400">Transfer times are payload-size calculations, not radio measurements. Protocol overhead, reliability and real network conditions are excluded. A grid cell summarizes multiple points and loses detail; entity-count reduction is not an accuracy measure. Terrain annotations use geometric thresholds and are not validated navigation decisions.</p>
        </>}
    </AnalysisDialog>;
}
