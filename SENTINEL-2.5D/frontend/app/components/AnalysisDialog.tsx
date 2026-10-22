"use client";
import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";

export default function AnalysisDialog({isOpen,onClose,title,subtitle,children}:{isOpen:boolean;onClose:()=>void;title:string;subtitle:string;children:ReactNode}) {
    useEffect(()=>{
        if(!isOpen) return;
        const previous=document.activeElement instanceof HTMLElement?document.activeElement:null;
        const escape=(event:KeyboardEvent)=>{if(event.key==="Escape")onClose();};
        document.addEventListener("keydown",escape);
        return ()=>{document.removeEventListener("keydown",escape);previous?.focus();};
    },[isOpen,onClose]);
    if(!isOpen)return null;
    return <div className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4 backdrop-blur-md">
        <section role="dialog" aria-modal="true" aria-label={title} className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-cyan-400/30 bg-[#0b1521] text-slate-100 shadow-2xl">
            <header className="flex items-start justify-between gap-4 border-b border-white/10 px-6 py-5">
                <div><h2 className="text-lg font-semibold">{title}</h2><p className="mt-1 text-xs leading-relaxed text-slate-400">{subtitle}</p></div>
                <button autoFocus onClick={onClose} aria-label="Close analysis" className="rounded-lg border border-white/15 p-2 hover:bg-white/10"><X className="h-4 w-4"/></button>
            </header>
            <div className="space-y-5 overflow-y-auto p-6">{children}</div>
            <footer className="flex justify-end border-t border-white/10 px-6 py-3"><button onClick={onClose} className="rounded-lg bg-cyan-300 px-4 py-2 text-sm font-semibold text-slate-950">Dismiss Analysis</button></footer>
        </section>
    </div>;
}
