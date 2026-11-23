"use client";

import React, { useEffect, useRef } from "react";
import Image from "next/image";
import gsap from "gsap";

export function PageTransition({ children }: { children: React.ReactNode }) {
  const contentRef = useRef<HTMLDivElement | null>(null);
  const curtainRef = useRef<HTMLDivElement | null>(null);
  const logoRef = useRef<HTMLDivElement | null>(null);
  const brandRef = useRef<HTMLDivElement | null>(null);
  const mottoRef = useRef<HTMLParagraphElement | null>(null);
  const hasPlayedInitialIntroRef = useRef(false);

  // Run once on full page load (RootLayout stays mounted across route/link changes)
  useEffect(() => {
    const curtain = curtainRef.current;
    if (!curtain) return;

    if (hasPlayedInitialIntroRef.current) {
      gsap.set(curtain, {
        autoAlpha: 0,
        pointerEvents: "none",
        display: "none",
      });
      return;
    }

    // Failsafe timer: guarantee curtain always clears within 3.8s even if tab was backgrounded
    const safetyTimer = window.setTimeout(() => {
      hasPlayedInitialIntroRef.current = true;
      if (curtainRef.current) {
        curtainRef.current.style.display = "none";
        curtainRef.current.style.pointerEvents = "none";
      }
    }, 3800);

    const enterDuration = 0.65;
    const holdDelay = 0.5;
    const liftDuration = 0.75;

    const ctx = gsap.context(() => {
      gsap.set(curtain, {
        yPercent: 0,
        autoAlpha: 1,
        display: "flex",
        pointerEvents: "auto",
      });

      const tl = gsap.timeline({
        onComplete: () => {
          hasPlayedInitialIntroRef.current = true;
          window.clearTimeout(safetyTimer);
          gsap.set(curtain, {
            autoAlpha: 0,
            pointerEvents: "none",
            display: "none",
          });
        },
      });

      tl.fromTo(
        logoRef.current,
        { scale: 0.78, opacity: 0, y: 18 },
        {
          scale: 1,
          opacity: 1,
          y: 0,
          duration: enterDuration,
          ease: "back.out(1.6)",
        }
      )
        .fromTo(
          brandRef.current,
          { opacity: 0, y: 12 },
          {
            opacity: 1,
            y: 0,
            duration: enterDuration * 0.8,
            ease: "power3.out",
          },
          "-=0.2"
        )
        .fromTo(
          mottoRef.current,
          { opacity: 0, y: 10 },
          {
            opacity: 1,
            y: 0,
            duration: enterDuration * 0.8,
            ease: "power3.out",
          },
          "-=0.16"
        )
        .set(curtain, { pointerEvents: "none" }, `+=${holdDelay}`)
        .to(curtain, {
          yPercent: -100,
          duration: liftDuration,
          ease: "power4.inOut",
        });

      if (contentRef.current) {
        gsap.fromTo(
          contentRef.current,
          { opacity: 0, y: 16 },
          {
            opacity: 1,
            y: 0,
            duration: 0.48,
            delay: enterDuration + holdDelay + 0.08,
            ease: "power3.out",
            clearProps: "transform",
          }
        );
      }
    });

    return () => {
      window.clearTimeout(safetyTimer);
      ctx.revert();
    };
  }, []);

  return (
    <>
      {/* First-Load Only Intro Curtain Overlay */}
      <div
        ref={curtainRef}
        aria-hidden="true"
        className="fixed inset-0 z-9995 flex flex-col items-center justify-center bg-linear-to-br from-[#e3f1ef] via-[#f1faf8] to-[#d4ebe7] px-6 select-none will-change-transform"
      >
        {/* Soft ambient teal/emerald glow behind logo */}
        <div className="pointer-events-none absolute size-72 rounded-full bg-[#0ea86b]/15 blur-3xl" />

        <div className="relative z-10 flex flex-col items-center text-center gap-5">
          <div
            ref={logoRef}
            className="relative size-28 sm:size-32 rounded-full overflow-hidden bg-white shadow-[0_20px_50px_rgba(11,85,99,0.2)] border-4 border-white ring-4 ring-[#0ea86b]/15"
          >
            <Image
              src="/logo.jpg"
              alt="OushudhCheck Logo"
              fill
              sizes="128px"
              className="object-cover"
              priority
            />
          </div>

          <div ref={brandRef} className="space-y-1">
            <p className="text-2xl sm:text-3xl font-extrabold tracking-[0.18em] uppercase text-[#09353c]">
              OUSHUDH<span className="text-[#0ea86b]">CHECK</span>
            </p>
          </div>

          <p
            ref={mottoRef}
            className="text-base sm:text-lg font-extrabold tracking-tight text-[#09353c] leading-snug"
          >
            Know Your Medicines.
            <br />
            <span className="bg-linear-to-r from-[#0b5563] via-[#0d9488] to-[#0ea86b] bg-clip-text text-transparent">
              Protect Your Health.
            </span>
          </p>
        </div>
      </div>

      {/* Page Content Container */}
      <div ref={contentRef} className="min-h-full flex flex-col flex-1">
        {children}
      </div>
    </>
  );
}
