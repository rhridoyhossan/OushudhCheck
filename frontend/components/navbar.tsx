"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Sparkles, Cpu } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: "/", label: "Overview" },
    { href: "/scan", label: "Scan Medicine" },
    { href: "/database", label: "Full Database" },
  ];

  const isPipelineActive = pathname === "/pipeline";

  return (
    <header className="w-full pt-6 pb-4 px-4 sm:px-8 max-w-6xl mx-auto flex items-center justify-between">
      {/* Left Logo using User's Uploaded Logo */}
      <Link href="/" className="flex items-center gap-2.5 group">
        <div className="relative size-10 rounded-xl overflow-hidden bg-white shadow-sm border border-white shrink-0">
          <Image
            src="/logo.jpg"
            alt="OushudhCheck Logo"
            fill
            sizes="40px"
            className="object-cover"
            priority
          />
        </div>
        <span className="font-extrabold text-base tracking-wider uppercase text-[#09353c]">
          OUSHUDH<span className="text-[#0ea86b]">CHECK</span>
        </span>
      </Link>

      {/* Center Minimal Links */}
      <nav className="hidden md:flex items-center gap-8 text-xs font-bold uppercase tracking-widest text-[#3b6970]">
        {navLinks.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`transition ${
                isActive
                  ? "text-[#0ea86b] font-extrabold underline decoration-[#0ea86b] decoration-2 underline-offset-8"
                  : "hover:text-[#09353c]"
              }`}
            >
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* Right Pill Buttons: Pipeline + Scan Now */}
      <div className="flex items-center gap-2.5">
        <Link
          href="/pipeline"
          className={`inline-flex items-center gap-1.5 rounded-full text-xs font-bold uppercase tracking-wider px-4 sm:px-5 py-2.5 shadow-sm border transition active:scale-95 group ${
            isPipelineActive
              ? "bg-[#0b5563] text-white border-[#0b5563]"
              : "bg-white/95 hover:bg-[#0b5563] text-[#09353c] hover:text-white border-white"
          }`}
        >
          <Cpu
            className={`size-3.5 transition ${
              isPipelineActive
                ? "text-[#6ee7b7]"
                : "text-[#0ea86b] group-hover:text-white"
            }`}
          />
          <span>Pipeline</span>
        </Link>

        <Link
          href="/scan"
          className="inline-flex items-center gap-1.5 rounded-full bg-white/95 hover:bg-[#0ea86b] text-[#09353c] hover:text-white text-xs font-bold uppercase tracking-wider px-5 py-2.5 shadow-sm border border-white transition active:scale-95 group"
        >
          <Sparkles className="size-3.5 text-[#0ea86b] group-hover:text-white transition" />
          <span>Scan Now</span>
        </Link>
      </div>
    </header>
  );
}
