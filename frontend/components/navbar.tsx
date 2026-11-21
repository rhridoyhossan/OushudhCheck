"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Sparkles, Cpu } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setIsScrolled(window.scrollY > 14);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navLinks = [
    { href: "/", label: "Overview" },
    { href: "/scan", label: "Scan Medicine" },
    { href: "/pipeline", label: "AI Pipeline Inspector" },
    { href: "/database", label: "Full Database" },
  ];

  return (
    <header className="sticky top-1.5 z-40 w-full px-4 sm:px-6 pt-1 pb-2 mb-5 sm:mb-7 pointer-events-none">
      <div
        className={`relative pointer-events-auto max-w-6xl mx-auto rounded-full px-5 sm:px-8 py-3 transition-all duration-300 overflow-hidden ${
          isScrolled
            ? "bg-white/55 backdrop-blur-3xl backdrop-saturate-150 border border-white/80 shadow-[0_14px_42px_rgba(11,85,99,0.14)]"
            : "bg-white/45 backdrop-blur-2xl backdrop-saturate-150 border border-white/70 shadow-[0_6px_24px_rgba(11,85,99,0.06)]"
        }`}
      >
        <div className="relative z-10 flex items-center justify-between gap-6">
          {/* Left Logo */}
          <Link href="/" className="flex items-center gap-2.5 group shrink-0">
            <div className="relative size-9 rounded-full overflow-hidden bg-white shadow-sm border border-white shrink-0">
              <Image
                src="/logo.jpg"
                alt="OushudhCheck Logo"
                fill
                sizes="36px"
                className="object-cover"
                priority
              />
            </div>
            <span className="font-extrabold text-sm sm:text-base tracking-wider uppercase text-[#09353c]">
              OUSHUDH<span className="text-[#0ea86b]">CHECK</span>
            </span>
          </Link>

          {/* Center Minimal Links */}
          <nav className="hidden md:flex items-center gap-9 lg:gap-11 text-xs font-bold uppercase tracking-widest text-[#3b6970]">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`transition flex items-center gap-1 ${
                    isActive
                      ? "text-[#0ea86b] font-extrabold underline decoration-[#0ea86b] decoration-2 underline-offset-8"
                      : "hover:text-[#09353c]"
                  }`}
                >
                  {link.href === "/pipeline" && (
                    <Cpu className="size-3.5 text-[#0ea86b]" />
                  )}
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Right Action Pill Button */}
          <Link
            href="/scan"
            className="inline-flex items-center gap-1.5 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs font-bold uppercase tracking-wider px-4 sm:px-5 py-2 shadow-sm transition active:scale-95 shrink-0"
          >
            <Sparkles className="size-3.5" />
            <span>Scan Now</span>
          </Link>
        </div>

        {/* Mobile Navigation Pills */}
        <nav className="relative z-10 flex md:hidden items-center justify-center gap-1.5 overflow-x-auto pt-2.5 mt-2 border-t border-teal-100/60">
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1 text-[11px] font-bold whitespace-nowrap transition ${
                  isActive
                    ? "bg-[#0ea86b] text-white shadow-xs"
                    : "bg-white/80 text-[#3b6970]"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
