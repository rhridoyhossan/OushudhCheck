"use client";

import React from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

export function HeroSection() {
  return (
    <section className="text-center max-w-3xl mx-auto px-4 pt-10 sm:pt-14 pb-12">
      <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#09353c] leading-tight">
        Know Your Medicines.
        <br />
        <span className="bg-gradient-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] bg-clip-text text-transparent">
          Protect Your Health.
        </span>
      </h1>

      <p className="mt-5 text-sm sm:text-base text-[#46737a] max-w-xl mx-auto leading-relaxed">
        AI-powered medicine identification, ingredient matching, interaction
        screening, and Bangla voice guidance in one platform.
      </p>
      <div className="mt-8 flex items-center justify-center gap-3">
        <Link
          href="/scan"
          className="inline-flex items-center gap-2 rounded-full bg-[#0b5563] hover:bg-[#0ea86b] text-white text-xs font-bold uppercase tracking-widest px-7 py-3.5 shadow-lg shadow-[#0b5563]/20 transition-all active:scale-95"
        >
          <span>Open Medicine Scanner</span>
          <ArrowUpRight className="size-4" />
        </Link>
      </div>
    </section>
  );
}
