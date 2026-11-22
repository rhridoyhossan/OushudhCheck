"use client";

import React, { useRef } from "react";
import { useParallaxZoom } from "@/components/scroll-reveal";
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
  Check,
  BookOpen,
} from "lucide-react";

export function ServicesGrid() {
  const sectionRef = useRef<HTMLElement | null>(null);
  useParallaxZoom(sectionRef, "[data-reveal='scale']");

  return (
    <section
      ref={sectionRef}
      id="services"
      className="relative max-w-5xl mx-auto px-4 sm:px-6 py-12"
    >
      {/* Frosted Glass Container with Perspective Grid & Ambient Glows (same as About section) */}
      <div className="relative rounded-[2.5rem] bg-white/60 backdrop-blur-2xl border border-white/90 px-5 py-12 sm:px-10 shadow-[0_24px_70px_rgba(11,85,99,0.09)] overflow-hidden">
        {/* Background Perspective Grid Lines in Teal */}
        <div
          className="pointer-events-none absolute inset-0 opacity-35"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(13,148,136,0.14) 1px, transparent 1px), linear-gradient(to bottom, rgba(13,148,136,0.14) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        {/* Soft Ambient Glow Orbs */}
        <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 size-80 rounded-full bg-[#0d9488]/18 blur-3xl" />
        <div className="pointer-events-none absolute top-1/3 -left-16 size-72 rounded-full bg-[#0ea86b]/16 blur-3xl" />
        <div className="pointer-events-none absolute bottom-10 -right-16 size-72 rounded-full bg-[#0b5563]/16 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-24 left-1/4 size-64 rounded-full bg-[#0ea86b]/15 blur-3xl" />

        {/* Section Header */}
        <div className="relative z-10 text-center mb-10">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-[#0ea86b]/10 text-[#0b5563] border border-[#0ea86b]/30 px-4 py-1 text-xs font-bold tracking-wide shadow-2xs mb-2.5">
            <Sparkles className="size-3.5 text-[#0ea86b]" />
            AI-Powered Healthcare Technology
          </span>
          <h2 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-[#09353c]">
            Our Services &amp;{" "}
            <span className="bg-gradient-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] bg-clip-text text-transparent">
              Capabilities
            </span>
          </h2>
        </div>

        {/* 3-Column Bento Grid with Glowing Card Borders & Hover Shadows (same glow effect as About section) */}
        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-3 gap-5 items-stretch">
          {/* ===============================================================
              COLUMN 1: Card 1 (AI Medicine Detection) + Card 3 (Duplicate Medicine Detection)
             =============================================================== */}
          <div className="flex flex-col gap-5">
            {/* CARD 1: 1. AI Medicine Detection */}
            <div data-reveal="scale" className="group flex-1 rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
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
                <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/70 space-y-2.5">
                  <div className="relative h-6 w-full rounded-full bg-[repeating-linear-gradient(120deg,#d5eae6,#d5eae6_4px,#f2f9f8_4px,#f2f9f8_8px)] flex items-center justify-between px-3">
                    <span className="text-[10px] font-semibold text-[#46737a] bg-white/90 px-2 py-0.5 rounded-full">
                      Rotated Strips
                    </span>
                    <span className="size-2.5 rounded-full bg-[#0d9488] ring-4 ring-white" />
                  </div>

                  <div className="relative h-7 w-full rounded-full bg-gradient-to-r from-[#0ea86b] to-[#0d9488] shadow-md shadow-[#0ea86b]/25 flex items-center justify-between px-3.5">
                    <span className="text-[11px] font-bold text-white">
                      YOLO26s-STAL
                    </span>
                    <span className="size-3 rounded-full bg-white ring-4 ring-emerald-300/50" />
                  </div>

                  <div className="relative h-7 w-3/4 rounded-full bg-gradient-to-r from-[#0b5563] to-[#0d9488] shadow-md shadow-[#0b5563]/20 flex items-center justify-between px-3.5">
                    <span className="text-[11px] font-bold text-white">
                      Foil PaddleOCR
                    </span>
                    <span className="size-3 rounded-full bg-white ring-4 ring-teal-300/50" />
                  </div>
                </div>

                {/* Thumbs-Up Callout Pill */}
                <div className="rounded-2xl bg-white p-3 shadow-[0_8px_25px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center gap-3">
                  <div className="flex items-center justify-center size-8 rounded-full bg-emerald-50 text-[#0ea86b] shrink-0 shadow-inner">
                    <ThumbsUp className="size-3.5 fill-[#0ea86b]" />
                  </div>
                  <p className="text-[11px] text-[#46737a] leading-snug">
                    Isolates <strong className="text-[#09353c]">98.4%</strong> of
                    rotated &amp; overlapping blister strips!
                  </p>
                </div>
              </div>

              <div className="mt-5 pt-2">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  1. YOLO26 Vision &amp; Foil OCR
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  AI Medicine Detection
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Detect multiple medicine strips from a single photo, even when
                  they are rotated, overlapping, or partially covered.
                  Automatically identify medicine names using AI-powered image
                  recognition and text extraction.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>YOLO26-powered detection</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Multi-strip scanning</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>OCR-based medicine identification</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* CARD 3: 3. Duplicate Medicine Detection */}
            <div data-reveal="scale" className="group rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
              <div className="space-y-2.5 overflow-hidden py-1">
                <div className="flex items-center gap-2.5">
                  <div className="flex-1 rounded-2xl bg-white p-3 shadow-[0_6px_20px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center gap-2.5">
                    <div className="flex items-center justify-center size-8 rounded-full bg-rose-100 text-rose-600 shrink-0">
                      <ShieldAlert className="size-4" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-bold text-[#09353c] truncate">
                        Duplicate Alert
                      </p>
                      <p className="text-[10px] text-[#46737a] truncate">
                        Napa + Ace ={" "}
                        <strong className="text-rose-600">Paracetamol</strong>
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl bg-[#f2f9f8] p-3 border border-teal-100/80 flex items-center gap-2 shrink-0">
                    <div className="flex items-center justify-center size-7 rounded-full bg-emerald-100 text-[#0ea86b]">
                      <CheckCircle2 className="size-3.5" />
                    </div>
                    <div>
                      <p className="text-[11px] font-bold text-[#09353c]">
                        Guard
                      </p>
                      <p className="text-[9px] text-[#0ea86b] font-semibold">
                        Active
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  3. Smart Duplicate Guard
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  Duplicate Medicine Detection
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Identify medicines that contain the same active ingredients,
                  even when they have different brand names. Help users avoid
                  accidentally taking duplicate medicines.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Brand-to-generic comparison</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Duplicate ingredient alerts</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Potential overdose warnings</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* ===============================================================
              COLUMN 2: Card 2 (Medicine Identification) + Card 4 (Drug Interaction Screening)
             =============================================================== */}
          <div className="flex flex-col gap-5">
            {/* CARD 2: 2. Medicine Identification & Generic Matching */}
            <div data-reveal="scale" className="group rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
              <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-[#46737a]">
                    BD Medicine Brands
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
                    <div className="absolute left-0 top-0 bottom-0 w-[76%] rounded-full bg-gradient-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] shadow-sm flex items-center justify-end pr-0.5">
                      <span className="size-3 rounded-full bg-white border-2 border-[#0ea86b] shadow" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-5">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  2. 21,000+ Bangladeshi Medicine Brands
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  Medicine Identification &amp; Generic Matching
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Identify medicines by their brand names and discover their
                  actual generic ingredients. Our matching system helps users
                  recognize when different brands contain the same active
                  ingredient.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>21,000+ brand records</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>1,700+ generic ingredients</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Intelligent text matching</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* CARD 4: 4. Drug Interaction Screening */}
            <div data-reveal="scale" className="group flex-1 rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
              <div className="space-y-3.5">
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
                        <strong className="text-[#09353c]">1,700+</strong>
                      </div>
                      <div>
                        <span className="text-[#46737a] block">Verified</span>
                        <strong className="text-[#0ea86b]">DrugBank</strong>
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
                  <div className="relative h-14 w-full pt-2 flex items-end justify-center overflow-hidden">
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
                    <div className="absolute bottom-0 left-[51%] -translate-x-1/2 h-11 w-3.5 rounded-t-full bg-gradient-to-b from-[#0ea86b] to-[#0b5563] shadow-md flex items-start justify-center pt-0.5">
                      <span className="size-2 rounded-full bg-white" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-5 pt-1">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  4. Drug-Drug Interaction (DDI) Analysis
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  Drug Interaction Screening
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Check whether the scanned medicines may interact with each
                  other. Highlight potential risks, contraindications, and
                  important precautions.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>DrugBank-based interaction screening</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Potential interaction alerts</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Medicine safety insights</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* ===============================================================
              COLUMN 3: Card 6 (Bangla AI Voice Assistant) + Card 7 (Telegram Medicine Assistant)
             =============================================================== */}
          <div className="flex flex-col gap-5">
            {/* CARD 6: 6. Bangla AI Voice Assistant */}
            <div data-reveal="scale" className="group flex-1 rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
              <div className="space-y-4">
                {/* Top 3 Mini Stat Squares */}
                <div className="grid grid-cols-3 gap-2.5">
                  <div className="rounded-2xl bg-white p-2.5 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                    <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0b5563] mb-1.5">
                      <Zap className="size-3" />
                    </div>
                    <p className="text-xs font-extrabold text-[#09353c]">
                      Gemma 3 1B
                    </p>
                    <p className="text-[9px] text-[#46737a]">SLM Engine</p>
                  </div>

                  <div className="rounded-2xl bg-white p-2.5 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                    <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0ea86b] mb-1.5">
                      <Volume2 className="size-3" />
                    </div>
                    <p className="text-xs font-extrabold text-[#09353c]">
                      bn-BD
                    </p>
                    <p className="text-[9px] text-[#46737a]">Bangla TTS</p>
                  </div>

                  <div className="rounded-2xl bg-white p-2.5 shadow-[0_6px_18px_rgba(11,85,99,0.06)] border border-teal-50">
                    <div className="flex items-center justify-center size-6 rounded-full bg-[#f2f9f8] text-[#0d9488] mb-1.5">
                      <Play className="size-3 fill-current" />
                    </div>
                    <p className="text-xs font-extrabold text-[#09353c]">
                      Audio
                    </p>
                    <p className="text-[9px] text-[#46737a]">Readout</p>
                  </div>
                </div>

                {/* Center Glowing Medical Plus Emblem with Concentric Teal Rings */}
                <div className="relative py-2 flex items-center justify-center">
                  <div className="size-28 rounded-full border border-teal-100 flex items-center justify-center">
                    <div className="size-20 rounded-full border border-emerald-200/80 bg-emerald-50/35 flex items-center justify-center">
                      <div className="size-13 rounded-full bg-white shadow-[0_10px_30px_rgba(14,168,107,0.28)] flex items-center justify-center">
                        <Plus className="size-7 text-[#0ea86b] stroke-[3.5]" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Audio Equalizer Bar Card */}
                <div className="rounded-2xl bg-white p-3.5 shadow-[0_8px_25px_rgba(11,85,99,0.06)] border border-teal-50 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-bold text-[#09353c]">
                      Bangla Voice Guidance
                    </p>
                    <span className="inline-block mt-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-[10px] font-bold text-[#0ea86b]">
                      বাংলায় সহজ ব্যাখ্যা
                    </span>
                  </div>

                  <div className="flex items-end gap-1.5 h-9">
                    <span className="w-2 h-5 rounded-full bg-teal-100" />
                    <span className="w-2 h-8 rounded-full bg-[#0ea86b] shadow-xs" />
                    <span className="w-2 h-9 rounded-full bg-[repeating-linear-gradient(120deg,#99d5c9,#99d5c9_2px,#f2f9f8_2px,#f2f9f8_4px)]" />
                    <span className="w-2 h-6 rounded-full bg-teal-100" />
                    <span className="w-2 h-8 rounded-full bg-[#0b5563]" />
                  </div>
                </div>
              </div>

              <div className="mt-5 pt-1">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  6. Powered by Gemma 3 1B SLM &amp; Bangla TTS
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  Bangla AI Voice Assistant
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Get medicine information and safety warnings explained in
                  simple Bangla. Listen instead of reading complicated medical
                  instructions.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Natural Bangla voice guidance</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>AI-powered explanations</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Accessible for everyone</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* CARD 7: 7. Telegram Medicine Assistant */}
            <div data-reveal="scale" className="group rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300 flex flex-col justify-between">
              <div className="relative h-32 flex items-center justify-center">
                <div className="size-24 rounded-full border border-dashed border-teal-200 flex items-center justify-center">
                  <div className="relative size-14 rounded-full bg-white shadow-[0_8px_25px_rgba(14,168,107,0.22)] border border-emerald-100 flex items-center justify-center text-[#0ea86b]">
                    <Bot className="size-6" />
                    <span className="absolute -top-1 -right-1 flex items-center justify-center size-4.5 rounded-full bg-[#0b5563] text-white text-[10px] font-bold shadow">
                      +
                    </span>
                  </div>
                </div>

                <div className="absolute left-1 top-3 -rotate-12 rounded-full bg-white px-2.5 py-1 shadow-md border border-teal-50 flex items-center gap-1 text-[10px] font-bold text-[#09353c]">
                  <Mic className="size-3 text-[#0b5563]" />
                  <span>Voice Query</span>
                </div>

                <div className="absolute right-1 top-2 rotate-12 rounded-full bg-white px-2.5 py-1 shadow-md border border-teal-50 flex items-center gap-1 text-[10px] font-bold text-[#09353c]">
                  <Pill className="size-3 text-[#0d9488]" />
                  <span>Strip Photo</span>
                </div>

                <div className="absolute right-2 bottom-1 -rotate-6 rounded-full bg-white px-2.5 py-1 shadow-md border border-teal-50 flex items-center gap-1 text-[10px] font-bold text-[#09353c]">
                  <Volume2 className="size-3 text-[#0ea86b]" />
                  <span>Bangla Reply</span>
                </div>
              </div>

              <div className="mt-3">
                <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-2.5 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                  7. AI-Powered Telegram Bot
                </span>
                <h3 className="text-lg font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition mt-1.5">
                  Telegram Medicine Assistant
                </h3>
                <p className="text-xs text-[#46737a] mt-1.5 leading-relaxed">
                  Access medicine safety features directly through Telegram.
                  Send medicine strip photos or Bangla voice messages and
                  receive helpful responses.
                </p>
                <ul className="mt-3 space-y-1.5 text-[11px] font-semibold text-[#0b5563]">
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Scan medicine photos</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Ask questions using voice</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    <span>Receive Bangla responses</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* ===============================================================
            CARD 5 (Full-Width Bento Feature Card with Same Glow Effect): 5. Smart Dosage & Medicine Guidance
           =============================================================== */}
        <div data-reveal="scale" className="group relative z-10 mt-5 rounded-[1.75rem] bg-white/95 backdrop-blur-xl p-6 sm:p-8 border border-teal-200/80 hover:border-[#0ea86b] shadow-[0_14px_40px_rgba(11,85,99,0.08)] hover:shadow-[0_20px_50px_rgba(14,168,107,0.18)] hover:-translate-y-0.5 transition-all duration-300">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            {/* Left 7 Cols: Feature #5 Text & Bullet Points */}
            <div className="lg:col-span-7 space-y-2.5">
              <span className="inline-block rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200/70 px-3 py-0.5 text-[10px] font-extrabold uppercase tracking-wider">
                5. Easy-to-Understand Medicine Information
              </span>
              <h3 className="text-xl font-extrabold text-[#09353c] group-hover:text-[#0ea86b] transition">
                Smart Dosage &amp; Medicine Guidance
              </h3>
              <p className="text-xs sm:text-sm text-[#46737a] leading-relaxed">
                Understand what each medicine is used for, how it works, and
                what precautions to take. Get complicated medical information
                explained in simple language.
              </p>
              <ul className="pt-2 grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs font-semibold text-[#0b5563]">
                <li className="flex items-center gap-2 rounded-xl bg-[#f2f9f8] px-3 py-2 border border-teal-100/70">
                  <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                  <span>Beginner-friendly explanations</span>
                </li>
                <li className="flex items-center gap-2 rounded-xl bg-[#f2f9f8] px-3 py-2 border border-teal-100/70">
                  <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                  <span>Dosage information from verified sources</span>
                </li>
                <li className="flex items-center gap-2 rounded-xl bg-[#f2f9f8] px-3 py-2 border border-teal-100/70">
                  <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                  <span>Side effects and safety precautions</span>
                </li>
              </ul>
            </div>

            {/* Right 5 Cols: Bento Visual Widget */}
            <div className="lg:col-span-5 rounded-2xl bg-[#f2f9f8] p-5 border border-teal-100/80 space-y-3.5">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 text-xs font-bold text-[#09353c]">
                  <BookOpen className="size-4 text-[#0ea86b]" />
                  Plain-Language Summary
                </span>
                <span className="rounded-full bg-white px-2.5 py-0.5 text-[10px] font-bold text-[#0ea86b] border border-teal-100">
                  17,191+ Indexed
                </span>
              </div>

              <div className="flex items-center justify-center gap-2 py-1">
                <span className="rounded-full bg-white px-3 py-1 text-[10px] font-semibold text-[#46737a] border border-teal-100 -rotate-3">
                  সকাল (Morning)
                </span>
                <span className="rounded-full bg-emerald-50 px-3.5 py-1 text-[11px] font-bold text-[#0ea86b] border border-emerald-200 shadow-2xs">
                  দুপুর (Noon)
                </span>
                <span className="rounded-full bg-white px-3 py-1 text-[10px] font-semibold text-[#46737a] border border-teal-100 rotate-3">
                  রাত (Night)
                </span>
              </div>

              <div className="rounded-xl bg-white p-3 border border-teal-50 text-[11px] text-[#46737a] leading-relaxed shadow-2xs">
                <strong className="text-[#09353c]">Napa (Paracetamol):</strong>{" "}
                Used to relieve pain and reduce fever. Never exceed the
                recommended dosage or combine with another Paracetamol brand.
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
