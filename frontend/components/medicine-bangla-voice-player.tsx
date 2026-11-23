"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { Check, Loader2, Pause, Play, Volume2 } from "lucide-react";
import { fetchBanglaTtsAudioBlob } from "@/lib/api";
import { BanglaVoiceWaveform } from "@/components/bangla-voice-waveform";

interface MedicineBanglaVoicePlayerProps {
  text: string;
  label?: string;
  onPlayStart?: () => void;
}

function formatAudioTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function MedicineBanglaVoicePlayer({
  text,
  label = "বাংলা ভয়েস প্লেয়ার (Bangla Voice Player)",
  onPlayStart,
}: MedicineBanglaVoicePlayerProps) {
  const cleanedText = (text || "").trim();

  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [failedKey, setFailedKey] = useState<string | null>(null);
  const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [audioCurrentTime, setAudioCurrentTime] = useState(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const autoPlayQueuedRef = useRef(false);
  const onPlayStartRef = useRef(onPlayStart);

  useEffect(() => {
    onPlayStartRef.current = onPlayStart;
  }, [onPlayStart]);

  const isLoading = Boolean(
    cleanedText && loadedKey !== cleanedText && failedKey !== cleanedText
  );
  const isReady = Boolean(
    cleanedText && loadedKey === cleanedText && audioBlobUrl
  );
  const isPlaying = Boolean(cleanedText && playingKey === cleanedText);

  const speakWithBrowserFallback = useCallback((script: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setPlayingKey(null);
      return;
    }
    onPlayStartRef.current?.();
    setPlayingKey(script);
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(script);
    utterance.lang = "bn-BD";
    utterance.rate = 0.95;
    utterance.onend = () => setPlayingKey(null);
    utterance.onerror = () => setPlayingKey(null);
    window.speechSynthesis.speak(utterance);
  }, []);

  useEffect(() => {
    if (!cleanedText) return;

    let cancelled = false;
    let createdObjUrl: string | null = null;
    const controller = new AbortController();
    autoPlayQueuedRef.current = false;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    fetchBanglaTtsAudioBlob(cleanedText, null, controller.signal)
      .then((blob) => {
        if (cancelled) return;
        const objUrl = URL.createObjectURL(blob);
        createdObjUrl = objUrl;
        setAudioBlobUrl(objUrl);

        const audio = new Audio(objUrl);
        audioRef.current = audio;
        setAudioEl(audio);
        setAudioDuration(0);
        setAudioCurrentTime(0);

        audio.onloadedmetadata = () => {
          if (!cancelled) {
            setAudioDuration(audio.duration || 0);
          }
        };
        audio.ontimeupdate = () => {
          if (!cancelled) {
            setAudioCurrentTime(audio.currentTime || 0);
          }
        };
        audio.onended = () => {
          if (!cancelled) {
            setPlayingKey(null);
            setAudioCurrentTime(audio.duration || 0);
          }
        };
        audio.onerror = () => {
          if (!cancelled) setPlayingKey(null);
        };

        setLoadedKey(cleanedText);

        if (autoPlayQueuedRef.current) {
          autoPlayQueuedRef.current = false;
          onPlayStartRef.current?.();
          setPlayingKey(cleanedText);
          audio.play().catch(() => setPlayingKey(null));
        }
      })
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === "AbortError") return;
        setFailedKey(cleanedText);
        if (autoPlayQueuedRef.current) {
          autoPlayQueuedRef.current = false;
          speakWithBrowserFallback(cleanedText);
        }
      });

    return () => {
      cancelled = true;
      autoPlayQueuedRef.current = false;
      controller.abort();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (createdObjUrl) {
        URL.revokeObjectURL(createdObjUrl);
      }
    };
  }, [cleanedText, speakWithBrowserFallback]);

  const handleTogglePlay = () => {
    if (typeof window === "undefined" || !cleanedText) return;

    if (isPlaying) {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      setPlayingKey(null);
      return;
    }

    if (isLoading) {
      autoPlayQueuedRef.current = true;
      return;
    }

    if (audioRef.current && isReady) {
      if (
        audioDuration > 0 &&
        audioRef.current.currentTime >= audioDuration - 0.1
      ) {
        audioRef.current.currentTime = 0;
        setAudioCurrentTime(0);
      }
      onPlayStartRef.current?.();
      setPlayingKey(cleanedText);
      audioRef.current.play().catch(() => setPlayingKey(null));
      return;
    }

    speakWithBrowserFallback(cleanedText);
  };

  if (!cleanedText) return null;

  const progressPct =
    audioDuration > 0
      ? Math.min(100, Math.round((audioCurrentTime / audioDuration) * 100))
      : 0;

  return (
    <div className="rounded-2xl bg-gradient-to-r from-[#09353c] via-[#0b5563] to-[#0d6e6e] p-3.5 sm:p-4 text-white shadow-md border border-teal-700/40 space-y-2.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <button
            type="button"
            onClick={handleTogglePlay}
            className={`flex items-center justify-center size-10 rounded-xl shadow-sm transition cursor-pointer shrink-0 ${
              isLoading
                ? "bg-white/15 text-emerald-300"
                : isPlaying
                ? "bg-white text-[#09353c] hover:bg-emerald-50"
                : "bg-[#0ea86b] text-white hover:bg-emerald-400"
            }`}
            aria-label={isPlaying ? "Pause Bangla Summary Audio" : "Play Bangla Summary Audio"}
          >
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : isPlaying ? (
              <Pause className="size-4 fill-current" />
            ) : (
              <Play className="size-4 fill-current ml-0.5" />
            )}
          </button>

          <div className="min-w-0 space-y-0.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-extrabold tracking-wide flex items-center gap-1.5 truncate">
                <Volume2 className="size-3.5 text-emerald-300 shrink-0" />
                <span className="truncate">{label}</span>
              </span>
              {isLoading ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/20 text-amber-200 border border-amber-300/30 px-2 py-0.5 text-[10px] font-bold shrink-0">
                  <Loader2 className="size-2.5 animate-spin" />
                  অডিও তৈরি হচ্ছে...
                </span>
              ) : isReady ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-400/20 text-emerald-200 border border-emerald-300/30 px-2 py-0.5 text-[10px] font-bold shrink-0">
                  <Check className="size-2.5 stroke-[3]" />
                  {isPlaying ? "চলছে" : "বাংলায় শুনুন"}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-white/15 text-teal-100 border border-white/20 px-2 py-0.5 text-[10px] font-bold shrink-0">
                  ব্রাউজার ভয়েস
                </span>
              )}
            </div>
            <p className="text-[10px] text-teal-100/85">
              {isLoading
                ? "সম্পূর্ণ বাংলা সারসংক্ষেপ অডিও প্রস্তুত হচ্ছে..."
                : isReady
                ? `মোট দৈর্ঘ্য: ${formatAudioTime(audioDuration)} • সম্পন্ন: ${progressPct}%`
                : "সম্পূর্ণ বাংলা সারসংক্ষেপ শুনতে প্লে বাটনে চাপুন"}
            </p>
          </div>
        </div>
      </div>

      {isReady && (
        <div className="space-y-1">
          <BanglaVoiceWaveform
            audioUrl={audioBlobUrl}
            audioElement={audioEl}
            isLoading={isLoading}
            onSeek={(sec) => setAudioCurrentTime(sec)}
          />
          <div className="flex items-center justify-between text-[10px] font-mono text-teal-100/85">
            <span>{formatAudioTime(audioCurrentTime)}</span>
            <span className="font-bold text-emerald-300">
              {progressPct}% Done
            </span>
            <span>
              {audioDuration > 0 ? formatAudioTime(audioDuration) : "--:--"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
