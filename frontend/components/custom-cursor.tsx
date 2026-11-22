"use client";

import React, { useEffect, useRef, useState, useSyncExternalStore } from "react";

type CursorMode = "default" | "interactive" | "heading" | "text";

function subscribeFinePointer(callback: () => void) {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia("(pointer: fine)");
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getFinePointerSnapshot() {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(pointer: fine)").matches;
}

function getServerSnapshot() {
  return false;
}

export function CustomCursor() {
  const dotWrapRef = useRef<HTMLDivElement | null>(null);
  const ringWrapRef = useRef<HTMLDivElement | null>(null);

  const enabled = useSyncExternalStore(
    subscribeFinePointer,
    getFinePointerSnapshot,
    getServerSnapshot
  );
  const [visible, setVisible] = useState(false);
  const [mode, setMode] = useState<CursorMode>("default");
  const [isPressed, setIsPressed] = useState(false);

  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;

    document.documentElement.classList.add("custom-cursor-active");

    let mouseX = -100;
    let mouseY = -100;
    let ringX = -100;
    let ringY = -100;
    let rafId = 0;

    const interactiveSelector =
      'a, button, [role="button"], select, label, summary, [data-cursor="pointer"]';
    const textInputSelector = "input, textarea, [contenteditable='true']";
    const headingSelector = "h1, h2, h3";

    const animateRing = () => {
      const dx = mouseX - ringX;
      const dy = mouseY - ringY;
      ringX += dx * 0.2;
      ringY += dy * 0.2;

      if (ringWrapRef.current) {
        ringWrapRef.current.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;
      }

      if (Math.abs(dx) > 0.2 || Math.abs(dy) > 0.2) {
        rafId = window.requestAnimationFrame(animateRing);
      } else {
        ringX = mouseX;
        ringY = mouseY;
        if (ringWrapRef.current) {
          ringWrapRef.current.style.transform = `translate3d(${ringX}px, ${ringY}px, 0)`;
        }
        rafId = 0;
      }
    };

    const ensureRingAnimating = () => {
      if (!rafId) {
        rafId = window.requestAnimationFrame(animateRing);
      }
    };

    const syncDot = (x: number, y: number) => {
      mouseX = x;
      mouseY = y;
      if (dotWrapRef.current) {
        dotWrapRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      }
      ensureRingAnimating();
    };

    const onMouseMove = (e: MouseEvent) => {
      syncDot(e.clientX, e.clientY);
      setVisible(true);

      // While dragging to select text, keep a clean text cursor and never interfere
      if (e.buttons === 1) {
        const selection = window.getSelection();
        if (selection && selection.toString().length > 0) {
          setMode("text");
          return;
        }
      }

      const target = e.target as HTMLElement | null;
      if (!target) {
        setMode("default");
        return;
      }

      if (target.closest(textInputSelector)) {
        setMode("text");
      } else if (target.closest(interactiveSelector)) {
        setMode("interactive");
      } else if (target.closest(headingSelector)) {
        setMode("heading");
      } else {
        setMode("default");
      }
    };

    const onMouseDown = (e: MouseEvent) => {
      syncDot(e.clientX, e.clientY);
      setIsPressed(true);
    };
    const onMouseUp = (e: MouseEvent) => {
      syncDot(e.clientX, e.clientY);
      setIsPressed(false);
    };
    const onMouseLeave = () => setVisible(false);
    const onMouseEnter = () => setVisible(true);

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("mousedown", onMouseDown, { passive: true });
    window.addEventListener("mouseup", onMouseUp, { passive: true });
    document.addEventListener("mouseleave", onMouseLeave);
    document.addEventListener("mouseenter", onMouseEnter);

    return () => {
      document.documentElement.classList.remove("custom-cursor-active");
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("mouseleave", onMouseLeave);
      document.removeEventListener("mouseenter", onMouseEnter);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [enabled]);

  if (!enabled) return null;

  const isTextMode = mode === "text";
  const isInteractive = mode === "interactive";
  const isHeading = mode === "heading";

  return (
    <>
      {/* Outer zero-size positioner for Ring (never shifts when size/scale changes) */}
      <div
        ref={ringWrapRef}
        aria-hidden="true"
        className="pointer-events-none select-none fixed top-0 left-0 z-[9998] will-change-transform"
      >
        <div
          className={`-translate-x-1/2 -translate-y-1/2 rounded-full border transition-[opacity,width,height,background-color,border-color,transform] duration-200 ease-out ${
            visible && !isTextMode ? "opacity-100" : "opacity-0"
          } ${
            isHeading
              ? "size-16 border-[#0ea86b]/70 bg-[#0ea86b]/12"
              : isInteractive
              ? "size-12 border-[#0ea86b] bg-[#0ea86b]/14"
              : "size-10 border-[#0b5563]/45 bg-white/10"
          } ${isPressed ? "scale-90" : "scale-100"}`}
        />
      </div>

      {/* Outer zero-size positioner for Center Dot (always dead-center on mouse pixel) */}
      <div
        ref={dotWrapRef}
        aria-hidden="true"
        className="pointer-events-none select-none fixed top-0 left-0 z-[9999] will-change-transform"
      >
        <div
          className={`-translate-x-1/2 -translate-y-1/2 transition-[opacity,width,height,border-radius,background-color,transform] duration-150 ease-out ${
            visible ? "opacity-100" : "opacity-0"
          } ${
            isTextMode
              ? "w-[2px] h-5 rounded-full bg-[#0ea86b] shadow-[0_0_8px_rgba(14,168,107,0.6)]"
              : isInteractive || isHeading
              ? "size-3 rounded-full bg-[#0ea86b] shadow-[0_0_12px_rgba(14,168,107,0.75)]"
              : "size-2.5 rounded-full bg-[#0b5563] shadow-[0_0_8px_rgba(11,85,99,0.35)]"
          } ${isPressed && !isTextMode ? "scale-75" : "scale-100"}`}
        />
      </div>
    </>
  );
}
