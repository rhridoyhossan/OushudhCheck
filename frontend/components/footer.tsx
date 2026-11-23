"use client";

import React, { useEffect, useRef } from "react";
import Image from "next/image";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Check } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

export function Footer() {
  const footerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = footerRef.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        el,
        { opacity: 0, y: 20 },
        {
          opacity: 1,
          y: 0,
          duration: 0.75,
          ease: "power2.out",
          clearProps: "transform,opacity",
          scrollTrigger: {
            trigger: el,
            start: "top 98%",
            once: true,
          },
        }
      );
    }, footerRef);

    return () => ctx.revert();
  }, []);

  return (
    <footer
      ref={footerRef}
      id="footer"
      className="max-w-5xl w-full mx-auto px-4 sm:px-6 pt-12 pb-8 text-[#46737a]"
    >
      {/* 4-Column Footer with Vertical Dashed Dividers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 pb-10 border-b border-teal-300/70">
        {/* Column 1: Contact + User's Uploaded Logo */}
        <div className="space-y-3 lg:pr-4">
          <h4 className="font-bold text-sm text-[#09353c]">Contact</h4>
          <div className="flex items-center gap-2.5 pt-1">
            <div className="relative size-9 rounded-xl overflow-hidden bg-white shadow-xs border border-white shrink-0">
              <Image
                src="/logo.jpg"
                alt="OushudhCheck Logo"
                fill
                className="object-cover"
              />
            </div>
            <p className="font-extrabold text-base text-[#09353c]">
              Oushudh<span className="text-[#0ea86b]">Check</span>
            </p>
          </div>
          <p className="text-xs text-[#46737a] leading-relaxed">
            Multi-Strip Medicine Safety Scanner &amp; Bangla Voice Assistant.
          </p>
        </div>

        {/* Column 2: Service */}
        <div className="space-y-2.5 lg:border-l lg:border-dashed lg:border-teal-300/80 lg:pl-6">
          <h4 className="font-bold text-sm text-[#09353c]">Service</h4>
          <ul className="space-y-2 text-xs text-[#46737a]">
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>YOLO26 Strip Vision</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>PaddleOCR Foil Reader</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>17K+ BD Medicine Index</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>Duplicate Overdose Check</span>
            </li>
          </ul>
        </div>

        {/* Column 3: Service */}
        <div className="space-y-2.5 lg:border-l lg:border-dashed lg:border-teal-300/80 lg:pl-6">
          <h4 className="font-bold text-sm text-[#09353c]">Service</h4>
          <ul className="space-y-2 text-xs text-[#46737a]">
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>DrugBank DDI Graph</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>Gemma 3 1B Bangla SLM</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>Bangla Voice Readout</span>
            </li>
            <li className="flex items-center gap-2">
              <Check className="size-3.5 text-[#0ea86b]" />
              <span>Telegram Voice Bot</span>
            </li>
          </ul>
        </div>

        {/* Column 4: Bangladesh Emergency Hotlines */}
        <div className="space-y-2.5 lg:border-l lg:border-dashed lg:border-teal-300/80 lg:pl-6">
          <h4 className="font-bold text-sm text-[#09353c]">
            Emergency Hotlines (BD)
          </h4>
          <div className="space-y-2 text-xs">
            <div>
              <p className="font-extrabold text-sm text-[#09353c]">
                16263: Shastho Batayon
              </p>
              <p className="text-[11px] text-[#46737a] leading-snug">
                24/7 National Health Helpline (DGHS Bangladesh)
              </p>
            </div>
            <div>
              <p className="font-extrabold text-sm text-[#09353c]">
                999: National Emergency
              </p>
              <p className="text-[11px] text-[#46737a] leading-snug">
                24/7 Ambulance, Fire &amp; Police Service (Bangladesh)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Copyright Bar */}
      <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#46737a]">
        <span>© 2026 OushudhCheck • AI Medicine Safety Scanner</span>
        <div className="flex items-center gap-4 font-semibold text-[#0b5563]">
          <span>Next.js</span>
          <span>YOLO26</span>
          <span>Bangla TTS</span>
        </div>
      </div>
    </footer>
  );
}
