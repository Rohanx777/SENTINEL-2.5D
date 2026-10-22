"use client";
import React,{useState} from "react";
import dynamic from "next/dynamic";
import {Radar,Layers,Box,Columns2,Activity,ChevronRight,Radio} from "lucide-react";
import {useLidarStream} from "./hooks/useLidarStream";
import TelemetryPanel from "./components/TelemetryPanel";
import ComparativeStudyModal from "./components/ComparativeStudyModal";
import FoveatedStratifiedModal from "./components/FoveatedStratifiedModal";
import type {VisualizationMode} from "./components/DeckCanvas2D";
const Canvas=dynamic(()=>import("./components/DeckCanvas2D"),{ssr:false,loading:()=>
    <div className="grid h-full place-items-center text-cyan-400">Initialising scene renderer…</div>});

export default function Dashboard(){
    const stream=useLidarStream();
    const [mode,setMode]=useState<VisualizationMode>("grid_25d");
    const [study,setStudy]=useState(false);const [zones,setZones]=useState(false);
    const counts=stream.cells.reduce<Record<string,number>>((a,c)=>{a[c.label]=(a[c.label]||0)+1;return a;},{});
    const status=stream.connected?stream.telemetry.engine:"WAITING FOR STREAM";
    return <main className="flex h-screen flex-col overflow-hidden bg-[#0b121b] text-slate-200">
        <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-[#111c29] px-5 py-4">
            <div className="flex items-center gap-3"><div className="rounded-xl border border-cyan-400/25 bg-cyan-400/10 p-2.5"><Radar className="h-6 w-6 text-cyan-300"/></div>
                <div><h1 className="text-lg font-semibold tracking-[.18em] text-white">SENTINEL <span className="text-cyan-300">2.5D</span></h1>
                    <p className="mt-1 text-[10px] uppercase tracking-[.16em] text-slate-500">LiDAR scene explorer · Research prototype</p></div></div>
            <div className="text-right"><div className="flex items-center justify-end gap-2 text-xs font-mono"><span className={`h-2 w-2 rounded-full ${stream.connected?'bg-emerald-400':'bg-amber-400'}`}/>{status}</div>
                <p className="mt-1 text-[10px] text-slate-500">Sensor coordinates · {stream.telemetry.mode || 'Awaiting data'}</p></div>
        </header>
        <nav className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-white/5 px-5 py-3">
            <div className="flex gap-1 rounded-lg bg-white/[.04] p-1">{[
                ['grid_25d','2.5D grid',Layers],['pointcloud_3d','3D preview',Box],['split_comparison','Compare',Columns2],
            ].map(([value,label,Icon])=>{const I=Icon as typeof Layers;return <button key={String(value)} onClick={()=>setMode(value as VisualizationMode)} className={`flex items-center gap-2 rounded-md px-3 py-2 text-xs ${mode===value?'bg-cyan-300/15 text-cyan-200':'text-slate-400 hover:text-white'}`}><I className="h-3.5 w-3.5"/>{String(label)}</button>;})}</div>
            <div className="flex items-center gap-2 text-xs"><span className="text-slate-500">Frame stride</span>{[1,2,4,8].map(speed=><button key={speed} onClick={()=>stream.setPlaybackSpeed(speed)} className={`rounded-md border px-2.5 py-1.5 font-mono ${stream.playbackSpeed===speed?'border-cyan-400/40 bg-cyan-400/10 text-cyan-200':'border-white/10 text-slate-400'}`}>{speed}×</button>)}</div>
        </nav>
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
            <section className="relative min-h-[320px] min-w-0 flex-1 overflow-hidden">
                <Canvas cells={stream.cells} linearCells={stream.linearCells} pointCloud3D={stream.pointCloud3D} egoVehicle={stream.egoVehicle}
                    threats={[]} gridType="polar" visMode={mode} coordinateFrame="sensor"/>
                {!stream.connected&&<div className="absolute inset-0 grid place-items-center bg-[#0b121b]/70"><div className="rounded-2xl border border-white/10 bg-[#111c29] p-6 text-center"><Radio className="mx-auto mb-3 h-7 w-7 text-cyan-300"/><p className="text-sm text-white">Waiting for perception stream</p><p className="mt-2 text-xs text-slate-400">Start SENTINEL in demo, replay, or ML mode.</p></div></div>}
                <div className="absolute bottom-4 left-4 flex flex-wrap gap-3 rounded-xl border border-white/10 bg-[#111c29]/95 px-4 py-3 text-[10px] text-slate-300">
                    <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#5ba38f]"/>Terrain</span><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#d3937e]"/>Static</span><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#e1bc6b]"/>Vehicle / person class</span><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#a797d1]"/>Geometry annotation</span>
                </div>
            </section>
            <aside className="max-h-[38vh] w-full shrink-0 overflow-y-auto border-l border-white/10 bg-[#111c29] p-4 lg:max-h-none lg:w-80">
                <TelemetryPanel metrics={stream.connected?stream.telemetry:null}/>
                <div className="mt-4 rounded-xl border border-white/10 p-4"><p className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400"><Activity className="h-3.5 w-3.5"/>Scene summary</p>
                    {['road','static_obstacle','dynamic_target','negative_obstacle','curb'].map(label=><div key={label} className="flex justify-between py-1.5 text-xs"><span className="text-slate-400">{label.replaceAll('_',' ')}</span><span className="font-mono text-white">{counts[label]||0} cells</span></div>)}
                    <p className="mt-3 border-t border-white/5 pt-3 text-[10px] leading-5 text-slate-500">Vehicle/person semantics do not establish motion or hostility. Terrain annotations use geometric thresholds.</p>
                </div>
                <div className="mt-4 rounded-xl border border-white/10 p-4 text-xs"><p className="text-slate-400">Total transmitted frame</p><p className="mt-2 font-mono text-lg text-cyan-200">{stream.connected&&stream.telemetry.payloadBytes?`${(stream.telemetry.payloadBytes/1024).toFixed(1)} KiB`:'—'}</p><p className="mt-1 text-[10px] text-slate-500">Includes preview, grid and metadata</p></div>
                <button onClick={()=>setStudy(true)} className="mt-4 flex w-full items-center justify-between rounded-xl border border-white/10 p-3 text-xs text-slate-300">Representation sizes<ChevronRight className="h-4 w-4"/></button>
                <button onClick={()=>setZones(true)} className="mt-2 flex w-full items-center justify-between rounded-xl border border-white/10 p-3 text-xs text-slate-300">Grid resolution and link estimates<ChevronRight className="h-4 w-4"/></button>
            </aside>
        </div>
        <footer className="flex shrink-0 justify-between border-t border-white/5 px-5 py-2 text-[10px] text-slate-500"><span>Measured processing metrics · CPU/CUDA method shown explicitly</span><span>Development build · Sensor frame</span></footer>
        <ComparativeStudyModal isOpen={study} onClose={()=>setStudy(false)} data={stream.comparativeStudy}/>
        <FoveatedStratifiedModal isOpen={zones} onClose={()=>setZones(false)} data={stream.stratifiedMetrics}/>
    </main>;
}
