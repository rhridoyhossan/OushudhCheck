"use client";

import { useEffect, type RefObject } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

/**
 * Runs a GSAP parallax scroll-zoom animation on elements owned by the calling component.
 * Because this hook runs inside the owning component's own `useEffect`, it is guaranteed
 * to execute strictly after that component has hydrated (zero SSR hydration mismatches).
 */
export function useParallaxZoom(
  containerRef: RefObject<HTMLElement | null>,
  selector?: string
) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    const ctx = gsap.context(() => {
      const targets = selector
        ? Array.from(container.querySelectorAll<HTMLElement>(selector))
        : [container];

      const viewportH = window.innerHeight;

      targets.forEach((el, idx) => {
        const rect = el.getBoundingClientRect();
        const isInitiallyVisible = rect.top < viewportH * 0.85;

        if (isInitiallyVisible) {
          gsap.fromTo(
            el,
            {
              scale: 0.95,
              y: 18,
              opacity: 0,
              transformOrigin: "center center",
            },
            {
              scale: 1,
              y: 0,
              opacity: 1,
              duration: 0.7,
              delay: idx * 0.06,
              ease: "power3.out",
              clearProps: "transform,opacity",
            }
          );
          return;
        }

        gsap.fromTo(
          el,
          {
            scale: 0.88,
            y: 48,
            opacity: 0.35,
            transformOrigin: "center center",
          },
          {
            scale: 1,
            y: 0,
            opacity: 1,
            ease: "none",
            scrollTrigger: {
              trigger: el,
              start: "top 96%",
              end: "top 60%",
              scrub: 0.75,
              invalidateOnRefresh: true,
            },
          }
        );
      });

      ScrollTrigger.refresh();
    }, containerRef);

    return () => {
      ctx.revert();
    };
  }, [containerRef, selector]);
}
