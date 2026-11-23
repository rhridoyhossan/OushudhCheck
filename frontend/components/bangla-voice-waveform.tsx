"use client";

import React, { memo, useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";

interface BanglaVoiceWaveformProps {
  audioUrl: string | null;
  audioElement: HTMLAudioElement | null;
  isLoading: boolean;
  onSeek?: (seconds: number) => void;
}

export const BanglaVoiceWaveform = memo(function BanglaVoiceWaveform({
  audioUrl,
  audioElement,
  isLoading,
  onSeek,
}: BanglaVoiceWaveformProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const onSeekRef = useRef(onSeek);

  useEffect(() => {
    onSeekRef.current = onSeek;
  }, [onSeek]);

  useEffect(() => {
    const container = containerRef.current;
    if (isLoading || !container || !audioUrl || !audioElement) return;

    const ws = WaveSurfer.create({
      container,
      media: audioElement,
      ...(audioElement.currentSrc || audioElement.src ? {} : { url: audioUrl }),
      height: 44,
      waveColor: "rgba(153, 246, 228, 0.38)",
      progressColor: "#34d399",
      cursorColor: "#ffffff",
      cursorWidth: 2,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      normalize: true,
      interact: true,
    });

    const unsubscribeInteraction = ws.on("interaction", (newTime: number) => {
      onSeekRef.current?.(newTime);
    });

    wsRef.current = ws;
    return () => {
      unsubscribeInteraction();
      ws.destroy();
      wsRef.current = null;
    };
  }, [audioUrl, audioElement, isLoading]);

  if (isLoading || !audioUrl) {
    return (
      <div className="h-[54px] w-full rounded-xl bg-white/10 border border-white/10 flex items-center justify-center px-3">
        <div className="w-full h-1.5 rounded-full bg-white/20 overflow-hidden">
          <div className="h-full w-1/3 rounded-full bg-emerald-300/80 animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-full min-h-[54px] rounded-xl bg-white/10 border border-white/15 px-2.5 py-1 cursor-pointer"
      aria-label="Bangla Voice Audio Waveform"
    />
  );
});
