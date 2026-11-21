"use client";

import React from "react";
import {
  ThumbsUp,
  Zap,
  Volume2,
  Play,
  Plus,
  Pill,
  ShieldAlert,
  Bot,
  Sparkles,
  CheckCircle2,
  Mic,
} from "lucide-react";

export function ServicesGrid() {
  return (
    <section id="services" className="max-w-5xl mx-auto px-4 sm:px-6 py-12">
      {/* Section Header */}
      <div className="text-center mb-10">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-white/85 backdrop-blur-md px-4 py-1 text-xs font-bold text-[#0ea86b] shadow-2xs border border-white mb-2.5">
          <Sparkles className="size-3.5" />
          4-Stage Clinical AI Pipeline
        </span>
        <h2 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-[#09353c]">
          Our Services &amp; Capabilities
        </h2>
        <p className="text-xs sm:text-sm text-[#46737a] mt-1.5 max-w-xl mx-auto">
          Interactive multi-strip vision, deterministic Bangladeshi drug resolution,
          and natural Bangla voice synthesis.
        </p>
      </div>

      {/* 3-Column Bento Grid Unified in Logo Teal & Emerald Palette */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
        {/* ===============================================================
            COLUMN 1: Tall Top Card + Compact Bottom Card
           =============================================================== */}
        <div className="flex flex-col gap-5">
          {/* CARD 1 (Top-Left Tall): YOLO26 Strip Detector Benchmark */}
          <div className="flex-1 rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="space-y-4">
              {/* Top Floating Metric Pill */}
              <div className="relative rounded-2xl bg-white p-4 shadow-[0_8px_28px_rgba(11,85,99,0.07)] border border-teal-50 flex items-center justify-between">
                <span className="text-sm font-bold text-[#09353c]">
                  YOLO26s Speed
                </span>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-[#0ea86b]">▲</span>
                  <span className="text-lg font-extrabold text-[#09353c]">
                    43.0%
                  </span>
                </div>
                <span className="absolute -bottom-2 left-1/2 -translate-x-1/2 size-3.5 rounded-full bg-white border-2 border-teal-200 shadow-2xs" />
              </div>

              {/* Horizontal Comparison Bars Widget */}
              <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/70 space-y-3">
                {/* Bar 1: Striped Lower */}
                <div className="relative h-7 w-full rounded-full bg-[repeating-linear-gradient(120deg,#d5eae6,#d5eae6_4px,#f2f9f8_4px,#f2f9f8_8px)] flex items-center justify-between px-3">
                  <span className="text-[11px] font-semibold text-[#46737a] bg-white/90 px-2 py-0.5 rounded-full">
                    YOLOv8
                  </span>
                  <span className="size-3 rounded-full bg-[#0d9488] ring-4 ring-white" />
                </div>

                {/* Bar 2: Striped Median */}
                <div className="relative h-7 w-3/4 rounded-full bg-[repeating-linear-gradient(120deg,#d5eae6,#d5eae6_4px,#f2f9f8_4px,#f2f9f8_8px)] flex items-center justify-between px-3">
                  <span className="text-[11px] font-semibold text-[#46737a] bg-white/90 px-2 py-0.5 rounded-full">
                    Faster-RCNN
                  </span>
                  <span className="size-3 rounded-full bg-[#0b5563] ring-4 ring-white" />
                </div>

                {/* Bar 3: Solid Emerald-Teal ("YOLO26s-STAL") */}
                <div className="relative h-8 w-full rounded-full bg-gradient-to-r from-[#0ea86b] to-[#0d9488] shadow-md shadow-[#0ea86b]/25 flex items-center justify-between px-3.5">
                  <span className="text-[11px] font-bold text-white">
                    YOLO26s-STAL
                  </span>
                  <span className="size-3.5 rounded-full bg-white ring-4 ring-emerald-300/50" />
                </div>

                {/* Bar 4: Solid Deep Teal ("PaddleOCR") */}
                <div className="relative h-8 w-2/3 rounded-full bg-gradient-to-r from-[#0b5563] to-[#0d9488] shadow-md shadow-[#0b5563]/20 flex items-center justify-between px-3.5">
                  <span className="text-[11px] font-bold text-white">
                    PaddleOCR
                  </span>
                  <span className="size-3.5 rounded-full bg-white ring-4 ring-teal-300/50" />
                </div>
              </div>

              {/* Thumbs-Up Callout Pill */}
              <div className="rounded-2xl bg-white p-3.5 shadow-[0_8px_25px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center gap-3">
                <div className="flex items-center justify-center size-9 rounded-full bg-emerald-50 text-[#0ea86b] shrink-0 shadow-inner">
                  <ThumbsUp className="size-4 fill-[#0ea86b]" />
                </div>
                <p className="text-[11px] text-[#46737a] leading-snug">
                  Isolates <strong className="text-[#09353c]">98.4%</strong> of
                  rotated &amp; occluded blister strips in sub-50ms ONNX!
                </p>
              </div>
            </div>

            <div className="mt-6 pt-2">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                YOLO26 Vision &amp; Foil OCR
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                NMS-free Small-Target-Aware Label Assignment isolates 2–6
                blister strips and rectifies curved metallic foil glare.
              </p>
            </div>
          </div>

          {/* CARD 2 (Bottom-Left Compact): Control Centre / Duplicate & DDI Shield */}
          <div className="rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="space-y-2.5 overflow-hidden py-1">
              <div className="flex items-center gap-2.5">
                <div className="flex-1 rounded-2xl bg-white p-3 shadow-[0_6px_20px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center gap-2.5">
                  <div className="flex items-center justify-center size-8 rounded-full bg-teal-100 text-[#0b5563] shrink-0">
                    <ShieldAlert className="size-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-[#09353c] truncate">
                      Duplicate Guard
                    </p>
                    <p className="text-[10px] text-[#46737a] truncate">
                      Napa + Ace ={" "}
                      <strong className="text-[#0d9488]">Paracetamol</strong>
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl bg-[#f2f9f8] p-3 border border-teal-100/80 flex items-center gap-2 shrink-0">
                  <div className="flex items-center justify-center size-7 rounded-full bg-emerald-100 text-[#0ea86b]">
                    <CheckCircle2 className="size-3.5" />
                  </div>
                  <div>
                    <p className="text-[11px] font-bold text-[#09353c]">Seclo</p>
                    <p className="text-[9px] text-[#0ea86b] font-semibold">
                      Safe PPI
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div className="rounded-2xl bg-[#f2f9f8] p-3 border border-teal-100/60">
                  <p className="text-[11px] font-bold text-[#09353c]">
                    DDI Screening
                  </p>
                  <p className="text-[10px] text-[#46737a] truncate mt-0.5">
                    Rolac 10mg{" "}
                    <strong className="text-[#0b5563]">Meal Rule</strong>
                  </p>
                </div>
                <div className="rounded-2xl bg-white p-3 shadow-xs border border-teal-50 flex items-center gap-2">
                  <div className="flex items-center justify-center size-7 rounded-lg bg-[#0b5563] text-white font-black text-xs shrink-0">
                    Rx
                  </div>
                  <div className="min-w-0">
                    <p className="text-[11px] font-bold text-[#09353c] truncate">
                      DrugBank
                    </p>
                    <p className="text-[10px] text-[#0ea86b] font-semibold truncate">
                      Verified Graph
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                Duplicate &amp; DDI Control Centre
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                Prevents accidental overdose duplication (Napa + Ace) and screens
                NSAID interactions at a glance.
              </p>
            </div>
          </div>
        </div>

        {/* ===============================================================
            COLUMN 2: Compact Top Card + Tall Bottom Card
           =============================================================== */}
        <div className="flex flex-col gap-5">
          {/* CARD 3 (Top-Middle Compact): 21K+ BD Brand Resolver Goals Bar */}
          <div className="rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-[#46737a]">
                  BD Generic Index
                </span>
                <span className="text-base font-extrabold text-[#09353c]">
                  21,000+
                </span>
              </div>

              <div className="rounded-2xl bg-white p-3 shadow-[0_6px_20px_rgba(11,85,99,0.06)] border border-teal-50 space-y-2">
                <div className="flex items-center justify-between text-[10px] font-semibold text-[#46737a]">
                  <span>1,700+ Generics</span>
                  <span>99.2% Match</span>
                </div>
                <div className="relative h-3.5 w-full rounded-full bg-teal-50 flex items-center px-1">
                  <div className="w-full flex items-center justify-between px-2">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <span
                        key={i}
                        className="size-1 rounded-full bg-teal-300"
                      />
                    ))}
                  </div>
                  <div className="absolute left-0 top-0 bottom-0 w-[74%] rounded-full bg-gradient-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] shadow-sm flex items-center justify-end pr-0.5">
                    <span className="size-3 rounded-full bg-white border-2 border-[#0ea86b] shadow" />
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                21K+ BD Brand Resolver
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                RapidFuzz Levenshtein &amp; phonetic indexing resolves noisy foil
                OCR strings to exact Bangladeshi generic compounds.
              </p>
            </div>
          </div>

          {/* CARD 4 (Bottom-Middle Tall): Dashboards / Clinical Telemetry & Dosage */}
          <div className="flex-1 rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="space-y-4">
              <div className="rounded-2xl bg-[#f2f9f8] p-5 border border-teal-100/80 text-center">
                <p className="text-2xl sm:text-3xl font-extrabold text-[#09353c] tracking-tight">
                  42 ms ONNX
                </p>
                <p className="text-[11px] text-[#46737a] mt-0.5">
                  Real-time multi-strip safety inference
                </p>

                <div className="mt-4 flex items-center justify-center gap-2">
                  <span className="rounded-full bg-white px-3 py-1 text-[10px] font-semibold text-[#46737a] border border-teal-100 -rotate-6">
                    সকাল (Morning)
                  </span>
                  <span className="rounded-full bg-emerald-50 px-3.5 py-1 text-[11px] font-bold text-[#0ea86b] border border-emerald-200 shadow-2xs">
                    দুপুর (Noon)
                  </span>
                  <span className="rounded-full bg-white px-3 py-1 text-[10px] font-semibold text-[#46737a] border border-teal-100 rotate-6">
                    রাত (Night)
                  </span>
                </div>
              </div>

              {/* Rating Point Barcode Tick Bar + Smooth Emerald Wave Widget */}
              <div className="rounded-2xl bg-white p-4 shadow-[0_8px_25px_rgba(11,85,99,0.06)] border border-teal-50 space-y-3">
                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-[10px] font-semibold text-[#46737a]">
                      DDI Checked Pairs
                    </p>
                    <p className="text-base font-extrabold text-[#09353c]">
                      1,228
                    </p>
                  </div>
                  <div className="rounded-xl bg-[#f2f9f8] px-2.5 py-1 border border-teal-100/70 flex items-center gap-3 text-[10px]">
                    <div>
                      <span className="text-[#46737a] block">Generics</span>
                      <strong className="text-[#09353c]">1,740</strong>
                    </div>
                    <div>
                      <span className="text-[#46737a] block">Verified</span>
                      <strong className="text-[#0ea86b]">100%</strong>
                    </div>
                  </div>
                  <span className="text-xs font-semibold text-[#46737a]">
                    /1,928
                  </span>
                </div>

                {/* Emerald Vertical Tick Barcode */}
                <div className="flex items-center justify-between gap-0.5 pt-1">
                  {Array.from({ length: 28 }).map((_, idx) => (
                    <span
                      key={idx}
                      className={`h-3.5 w-1.5 rounded-xs ${
                        idx < 19 ? "bg-[#0ea86b]" : "bg-teal-100"
                      }`}
                    />
                  ))}
                </div>

                {/* Smooth Wave Curve + Teal-Emerald Pillar Indicator */}
                <div className="relative h-16 w-full pt-2 flex items-end justify-center overflow-hidden">
                  <svg
                    viewBox="0 0 300 70"
                    className="w-full h-full overflow-visible"
                  >
                    <defs>
                      <linearGradient
                        id="waveFillTeal"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#0ea86b"
                          stopOpacity="0.25"
                        />
                        <stop
                          offset="100%"
                          stopColor="#0ea86b"
                          stopOpacity="0.0"
                        />
                      </linearGradient>
                    </defs>
                    <path
                      d="M 0 60 Q 45 60, 75 45 T 155 15 T 230 45 T 300 35 L 300 70 L 0 70 Z"
                      fill="url(#waveFillTeal)"
                    />
                    <path
                      d="M 0 60 Q 45 60, 75 45 T 155 15 T 230 45 T 300 35"
                      fill="none"
                      stroke="#0d9488"
                      strokeWidth="2"
                    />
                  </svg>
                  <div className="absolute bottom-0 left-[51%] -translate-x-1/2 h-12 w-3.5 rounded-t-full bg-gradient-to-b from-[#0ea86b] to-[#0b5563] shadow-md flex items-start justify-center pt-0.5">
                    <span className="size-2 rounded-full bg-white" />
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-2">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                Clinical Safety Dashboards
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                Inspect resolved generic ingredients, meal-timing schedules, and
                DrugBank contraindications in one place.
              </p>
            </div>
          </div>
        </div>

        {/* ===============================================================
            COLUMN 3: Tall Top Card + Compact Bottom Card
           =============================================================== */}
        <div className="flex flex-col gap-5">
          {/* CARD 5 (Top-Right Tall): Bangla SLM & Voice Readout Insights */}
          <div className="flex-1 rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="space-y-5">
              {/* Top 3 Mini Stat Squares */}
              <div className="grid grid-cols-3 gap-2.5">
                <div className="rounded-2xl bg-white p-3 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                  <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0b5563] mb-2">
                    <Zap className="size-3" />
                  </div>
                  <p className="text-sm font-extrabold text-[#09353c]">1.5B</p>
                  <p className="text-[9px] text-[#46737a] mt-0.5">
                    Qwen2.5 SLM
                  </p>
                </div>

                <div className="rounded-2xl bg-white p-3 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                  <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0ea86b] mb-2">
                    <Volume2 className="size-3" />
                  </div>
                  <p className="text-sm font-extrabold text-[#09353c]">bn-BD</p>
                  <p className="text-[9px] text-[#46737a] mt-0.5">
                    Bangla Voice
                  </p>
                </div>

                <div className="rounded-2xl bg-white p-3 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                  <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0d9488] mb-2">
                    <Play className="size-3 fill-current" />
                  </div>
                  <p className="text-sm font-extrabold text-[#09353c]">4-Bit</p>
                  <p className="text-[9px] text-[#46737a] mt-0.5">
                    Local GGUF
                  </p>
                </div>
              </div>

              {/* Center Glowing Medical Plus Emblem with Concentric Teal Rings */}
              <div className="relative py-4 flex items-center justify-center">
                <div className="size-32 rounded-full border border-teal-100 flex items-center justify-center">
                  <div className="size-24 rounded-full border border-emerald-200/80 bg-emerald-50/35 flex items-center justify-center">
                    <div className="size-16 rounded-full bg-white shadow-[0_10px_30px_rgba(14,168,107,0.28)] flex items-center justify-center">
                      <Plus className="size-8 text-[#0ea86b] stroke-[3.5]" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Audio Equalizer Bar Card */}
              <div className="rounded-2xl bg-white p-4 shadow-[0_8px_25px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-bold text-[#09353c]">
                    Bangla TTS Accuracy
                  </p>
                  <span className="inline-block mt-1.5 rounded-full bg-emerald-50 px-2.5 py-0.5 text-[10px] font-bold text-[#0ea86b]">
                    99.4% Grounded
                  </span>
                </div>

                {/* Vertical Mini Bars in Logo Teal & Emerald */}
                <div className="flex items-end gap-1.5 h-11">
                  <span className="w-2 h-6 rounded-full bg-teal-100" />
                  <span className="w-2 h-9 rounded-full bg-[#0ea86b] shadow-xs" />
                  <span className="w-2 h-11 rounded-full bg-[repeating-linear-gradient(120deg,#99d5c9,#99d5c9_2px,#f2f9f8_2px,#f2f9f8_4px)]" />
                  <span className="w-2 h-7 rounded-full bg-teal-100" />
                  <span className="w-2 h-10 rounded-full bg-[#0b5563]" />
                </div>
              </div>
            </div>

            <div className="mt-6 pt-2">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                Bangla Voice &amp; SLM Readout
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                Fine-tuned Qwen2.5-1.5B synthesizes grounded dosage schedules and
                speaks warnings aloud in clear Bangla.
              </p>
            </div>
          </div>

          {/* CARD 6 (Bottom-Right Compact): Forecast / Omnichannel Telegram Bot Orbit */}
          <div className="rounded-[2rem] bg-white/95 backdrop-blur-xl p-6 border border-white shadow-[0_14px_40px_rgba(11,85,99,0.07)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.14)] transition-all flex flex-col justify-between">
            <div className="relative h-36 flex items-center justify-center">
              <div className="size-28 rounded-full border border-dashed border-teal-200 flex items-center justify-center">
                <div className="relative size-16 rounded-full bg-white shadow-[0_8px_25px_rgba(14,168,107,0.22)] border border-emerald-100 flex items-center justify-center text-[#0ea86b]">
                  <Bot className="size-7" />
                  <span className="absolute -top-1 -right-1 flex items-center justify-center size-5 rounded-full bg-[#0b5563] text-white text-[10px] font-bold shadow">
                    +
                  </span>
                </div>
              </div>

              {/* Floating Orbit Pill 1: Left */}
              <div className="absolute left-1 top-4 -rotate-12 rounded-full bg-white px-3 py-1 shadow-md border border-teal-50 flex items-center gap-1.5 text-[10px] font-bold text-[#09353c]">
                <Mic className="size-3 text-[#0b5563]" />
                <span>Whisper Voice</span>
              </div>

              {/* Floating Orbit Pill 2: Top Right */}
              <div className="absolute right-1 top-3 rotate-12 rounded-full bg-white px-3 py-1 shadow-md border border-teal-50 flex items-center gap-1.5 text-[10px] font-bold text-[#09353c]">
                <Pill className="size-3 text-[#0d9488]" />
                <span>Strip Photo</span>
              </div>

              {/* Floating Orbit Pill 3: Bottom Right */}
              <div className="absolute right-2 bottom-2 -rotate-6 rounded-full bg-white px-3 py-1 shadow-md border border-teal-50 flex items-center gap-1.5 text-[10px] font-bold text-[#09353c]">
                <Volume2 className="size-3 text-[#0ea86b]" />
                <span>Bangla Reply</span>
              </div>
            </div>

            <div className="mt-4">
              <h3 className="text-lg font-extrabold text-[#09353c]">
                Telegram Voice Bot
              </h3>
              <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                Send a strip photo or ask follow-up questions via Bangla voice
                notes on our asynchronous Telegram bot.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
