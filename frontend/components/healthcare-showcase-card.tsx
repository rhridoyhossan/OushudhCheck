"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import gsap from "gsap";
import { useParallaxZoom } from "@/components/scroll-reveal";
import {
  Volume2,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  Sparkles,
  Camera,
  Stethoscope,
  Database,
} from "lucide-react";

export function HealthcareShowcaseCard() {
  const containerRef = useRef<HTMLElement | null>(null);
  useParallaxZoom(containerRef);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const badges = gsap.utils.toArray<HTMLElement>(".floating-mini-badge");
      badges.forEach((badge, i) => {
        const yOffset = i % 2 === 0 ? -14 : -11;
        const xOffset = i % 2 === 0 ? 5 : -5;
        const rot = i % 2 === 0 ? 6 : -6;
        gsap.to(badge, {
          y: yOffset,
          x: xOffset,
          rotation: rot,
          duration: 2.2 + i * 0.35,
          repeat: -1,
          yoyo: true,
          ease: "sine.inOut",
          delay: i * 0.25,
        });
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={containerRef}
      data-reveal="scale"
      className="relative max-w-5xl mx-auto px-4 sm:px-6 pt-4 pb-12"
    >
      {/* Top Floating Circular Avatar / Badge Accents */}
      <div className="floating-mini-badge hidden sm:flex items-center justify-center size-11 rounded-full bg-white/90 backdrop-blur-md shadow-md border border-white absolute -top-1 left-1/4 z-20 text-[#0b5563] will-change-transform">
        <Stethoscope className="size-4.5 text-[#0ea86b]" />
      </div>
      <div className="floating-mini-badge hidden sm:flex items-center justify-center size-12 rounded-full bg-white/95 backdrop-blur-md shadow-md border-2 border-white absolute -top-2 right-1/4 z-20 overflow-hidden will-change-transform">
        <Image
          src="/logo.jpg"
          alt="OushudhCheck Badge"
          fill
          className="object-cover"
        />
      </div>

      {/* Main Frosted Glass Card with Logo Teal/Emerald Ambient Orbs */}
      <div className="relative rounded-[2.5rem] bg-white/65 backdrop-blur-2xl border border-white/90 p-6 sm:p-10 shadow-[0_24px_70px_rgba(11,85,99,0.12)] overflow-hidden">
        {/* Soft Blurred Teal & Emerald Orbs inside the Glass Card */}
        <div className="pointer-events-none absolute top-14 -left-10 size-52 rounded-full bg-[#0b5563]/35 blur-3xl" />
        <div className="pointer-events-none absolute top-8 right-10 size-44 rounded-full bg-[#0ea86b]/20 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-10 right-16 size-56 rounded-full bg-[#0d9488]/25 blur-3xl" />

        {/* Inner Top Mini Header Bar with User's Uploaded Logo */}
        <div className="relative z-10 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="relative size-8 rounded-lg overflow-hidden bg-white shadow-xs border border-white shrink-0">
              <Image
                src="/logo.jpg"
                alt="OushudhCheck Logo"
                fill
                className="object-cover"
              />
            </div>
            <span className="font-bold text-sm text-[#09353c]">
              OushudhCheck
            </span>
          </div>

          {/* Inner Mini Navigation Links */}
          <div className="flex items-center gap-6 text-xs font-semibold text-[#46737a]">
            <Link
              href="/"
              className="text-[#09353c] font-extrabold underline decoration-[#0ea86b] decoration-2 underline-offset-4"
            >
              Overview
            </Link>
            <Link href="/scan" className="hover:text-[#09353c] transition">
              YOLO26 Scanner
            </Link>
            <Link href="/database" className="hover:text-[#09353c] transition">
              17K+ BD Drugs
            </Link>
          </div>

          {/* Inner Top-Right Emerald Pill CTA */}
          <Link
            href="/scan"
            className="rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-[11px] font-bold uppercase tracking-wider px-5 py-2 shadow-md shadow-[#0ea86b]/25 transition"
          >
            Try Scanner
          </Link>
        </div>

        {/* Floating Right Circular Icon inside Card */}
        <div className="floating-mini-badge hidden md:flex items-center justify-center size-11 rounded-full bg-white/80 backdrop-blur-md shadow-sm border border-white absolute top-24 right-12 z-20 text-[#0b5563] will-change-transform">
          <Sparkles className="size-4 text-[#0ea86b]" />
        </div>

        {/* Center Giant Display Title: "Healthcare" */}
        <div className="relative z-10 text-center my-8 sm:my-10">
          <h2 className="text-5xl sm:text-7xl md:text-8xl font-extrabold tracking-tight text-[#09353c] select-none">
            Healthcare
          </h2>
        </div>

        {/* Floating Connected Mini Cards Layout */}
        <div className="relative z-10 grid grid-cols-1 md:grid-cols-12 gap-4 items-center my-4">
          {/* Left Floating Connected Mini Cards (White Card + Deep Teal Card) */}
          <div className="md:col-span-6 flex items-center justify-center md:justify-start gap-3">
            {/* White Mini Stat Card */}
            <div className="rounded-2xl bg-white/95 backdrop-blur-md p-4 shadow-md border border-white w-36">
              <p className="text-xs font-bold text-[#09353c]">YOLO26s</p>
              <p className="text-[11px] font-mono text-[#46737a] mt-0.5">
                4 Strips • 42ms
              </p>
              <div className="flex items-center gap-1.5 mt-3">
                <span className="size-2.5 rounded-full bg-[#0b5563]" />
                <span className="size-2.5 rounded-full bg-[#0ea86b] animate-pulse" />
                <span className="text-[10px] font-bold text-[#0d9488] ml-1">
                  Verified!
                </span>
              </div>
            </div>

            {/* Connector line */}
            <div className="h-0.5 w-4 bg-[#0b5563]/40 hidden sm:block" />

            {/* Deep Teal Connected Card */}
            <div className="rounded-2xl bg-[#0b5563] text-white p-4 shadow-lg w-44">
              <div className="flex items-center gap-1.5 text-xs font-bold text-[#6ee7b7]">
                <AlertTriangle className="size-3.5" />
                <span>Napa + Ace 500mg</span>
              </div>
              <p className="text-[11px] text-teal-100/90 mt-1 leading-snug">
                Duplicate Paracetamol detected. Overdose guard active.
              </p>
            </div>
          </div>

          {/* Right Floating Glass Info Card */}
          <div className="md:col-span-6 flex justify-center md:justify-end">
            <div className="rounded-2xl bg-white/85 backdrop-blur-md p-4 shadow-md border border-white max-w-xs w-full text-left">
              <div className="flex items-center justify-between">
                <p className="text-xs font-bold text-[#09353c]">
                  বাংলা ভয়েস সতর্কতা (Bangla AI)
                </p>
                <span className="text-[10px] font-mono text-[#0ea86b] font-bold">
                  Gemma 3 1B
                </span>
              </div>
              <p className="text-[11px] text-[#46737a] mt-1.5 line-clamp-2 leading-relaxed">
                “নাপা এবং এইস—দুটোই প্যারাসিটামল ৫০০ মিলিগ্রাম। ভুলেও এই দুটি ওষুধ একসাথে খাবেন না...”
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Centered Floating Search / Action Pill Bar */}
        <div className="relative z-10 mt-8 flex justify-center">
          <div className="flex items-center justify-between gap-3 rounded-full bg-white/95 backdrop-blur-xl px-5 py-2.5 shadow-lg border border-white w-full max-w-md">
            <Link
              href="/scan"
              className="flex items-center gap-2 text-xs font-bold text-[#09353c] hover:text-[#0ea86b] transition"
            >
              <Camera className="size-4 text-[#0ea86b]" />
              <span>Scan Strips</span>
            </Link>

            <div className="h-px flex-1 bg-teal-200" />

            <Link
              href="/database"
              className="flex items-center gap-1.5 text-xs font-semibold text-[#46737a] hover:text-[#09353c] transition"
            >
              <Database className="size-3.5 text-[#0d9488]" />
              <span>21K+ BD Database</span>
            </Link>

            <Link
              href="/scan"
              title="Go to Scanner"
              className="flex items-center justify-center size-9 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white shadow-md shadow-[#0ea86b]/30 transition"
            >
              <ArrowRight className="size-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Bottom Floating Circular Badges outside card */}
      <div className="floating-mini-badge hidden sm:flex items-center justify-center size-10 rounded-full bg-white/90 shadow-md border border-white absolute bottom-8 left-4 z-20 text-[#0b5563] will-change-transform">
        <ShieldCheck className="size-4 text-[#0ea86b]" />
      </div>
      <div className="floating-mini-badge hidden sm:flex items-center justify-center size-11 rounded-full bg-white/90 shadow-md border border-white absolute bottom-4 right-24 z-20 text-[#0b5563] will-change-transform">
        <Volume2 className="size-4 text-[#0d9488]" />
      </div>
    </section>
  );
}
