"use client";

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function SmoothScroll() {
  const pathname = usePathname();
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    // Respect reduced-motion accessibility preference
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (prefersReducedMotion) return;

    const lenis = new Lenis({
      lerp: 0.1,
      smoothWheel: true,
      wheelMultiplier: 0.95,
      touchMultiplier: 1.4,
      autoResize: true,
    });
    lenisRef.current = lenis;
    lenis.on("scroll", ScrollTrigger.update);

    let rafId = 0;
    function raf(time: number) {
      lenis.raf(time);
      rafId = window.requestAnimationFrame(raf);
    }
    rafId = window.requestAnimationFrame(raf);

    // Recompute Lenis scroll bounds (debounced) whenever page content dynamically grows/shrinks
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    const resizeObserver = new ResizeObserver(() => {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        lenis.resize();
        ScrollTrigger.refresh();
      }, 150);
    });
    resizeObserver.observe(document.body);

    // Automatically pause/resume Lenis when a modal locks body overflow so scroll never gets stuck
    const syncBodyLock = () => {
      const isLocked = document.body.style.overflow === "hidden";
      if (isLocked) {
        lenis.stop();
      } else {
        lenis.start();
        lenis.resize();
      }
    };
    const mutationObserver = new MutationObserver(syncBodyLock);
    mutationObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["style"],
    });
    syncBodyLock();

    return () => {
      window.cancelAnimationFrame(rafId);
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      lenis.destroy();
      lenisRef.current = null;
    };
  }, []);

  // Reset scroll cleanly on route navigation so Lenis never gets stuck at a stale offset
  useEffect(() => {
    const lenis = lenisRef.current;
    if (!lenis) return;
    document.body.style.overflow = "";
    lenis.start();
    lenis.scrollTo(0, { immediate: true, force: true });
    lenis.resize();
  }, [pathname]);

  return null;
}
