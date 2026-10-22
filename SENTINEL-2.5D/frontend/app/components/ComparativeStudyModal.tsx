"use client";
import AnalysisDialog from "./AnalysisDialog";
import type { ComparativeStudyData } from "../hooks/useLidarStream";

export default function ComparativeStudyModal({isOpen,onClose,data}:{isOpen:boolean;onClose:()=>void;data:ComparativeStudyData|null}) {
    const raw=data?.point_cloud_3d,grid=data?.grid_25d;
    return <AnalysisDialog isOpen={isOpen} onClose={onClose} title="Representation sizes" subtitle="Sizes from the current stream frame. Synthetic-demo values describe fixture data, not model accuracy.">
        {!data||!raw||!grid?<p className="text-sm text-slate-400">Waiting for stream data.</p>:<>
            <div className="grid gap-3 sm:grid-cols-3">
                {[["Raw float32 scan",raw.raw_size_kb.toFixed(1)+" KiB"],["JSON height grid",grid.raw_size_kb.toFixed(1)+" KiB"],["Grid byte reduction",data.summary.compression_ratio_pct.toFixed(1)+"%"]].map(([label,value])=><div key={label} className="rounded-xl border border-white/10 bg-white/[0.03] p-4"><p className="text-xs text-slate-400">{label}</p><p className="mt-2 font-mono text-2xl text-cyan-200">{value}</p></div>)}
            </div>
            <div className="overflow-x-auto rounded-xl border border-white/10">
                <table className="w-full text-left text-sm"><thead className="bg-white/5 text-slate-400"><tr><th className="p-3">Representation</th><th className="p-3">Elements</th><th className="p-3">Contents</th></tr></thead>
                <tbody className="divide-y divide-white/10"><tr><td className="p-3">Raw scan</td><td className="p-3">{raw.count.toLocaleString()} points</td><td className="p-3">x, y, z and intensity per point</td></tr><tr><td className="p-3">2.5D grid</td><td className="p-3">{grid.count.toLocaleString()} cells</td><td className="p-3">Aggregated height range and labels</td></tr></tbody></table>
            </div>
            <div className="rounded-xl border border-white/10 p-4 text-sm"><h3 className="font-medium">Illustrative transmission estimate</h3><p className="mt-2 text-slate-300">At an assumed 10 Mbps: {raw.estimated_tx_time_ms.toFixed(1)} ms for raw bytes and {grid.estimated_tx_time_ms.toFixed(1)} ms for the JSON grid.</p><p className="mt-2 text-xs leading-relaxed text-slate-400">Calculated from payload size only. Network headers, contention, retransmissions, propagation and browser time are excluded. This is not a measured network latency.</p></div>
            <p className="text-xs leading-relaxed text-slate-400">The complete WebSocket frame also includes the sampled preview and metadata; its total size is shown in dashboard telemetry. GPU memory and rendering throughput have not been measured here. Aggregation loses vertical detail and does not establish equivalent obstacle accuracy.</p>
        </>}
    </AnalysisDialog>;
}
