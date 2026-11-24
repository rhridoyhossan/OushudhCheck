"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { AITransparencyInspector } from "@/components/ai-transparency-inspector";
import { BanglaVoiceWaveform } from "@/components/bangla-voice-waveform";
import { analyzeMedicineImageStream, fetchBanglaTtsAudioBlob } from "@/lib/api";
import {
  BoundingBox,
  MedicineDatabaseRecord,
  ScanAnalysisResult,
} from "@/lib/types";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Building2,
  Check,
  Clock,
  Cpu,
  Database,
  ImagePlus,
  Loader2,
  Pause,
  Pill,
  Play,
  Sparkles,
  Stethoscope,
  Trash2,
  Upload,
  Volume2,
  X,
} from "lucide-react";
import { toast } from "sonner";

const PIPELINE_STAGE_STEPS = [
  {
    step: 1,
    title: "YOLO26s ONNX Object Detection",
    titleBn: "ধাপ ১/৬: YOLO26s ONNX স্ট্রিপ ডিটেকশন",
    desc: "Locating 1–6 blister pack bounding boxes via ONNXRuntime CPUExecutionProvider",
  },
  {
    step: 2,
    title: "OpenCV LAB-CLAHE Foil Preprocessing",
    titleBn: "ধাপ ২/৬: ফয়েল গ্লেয়ার রিমুভাল ও শার্পেনিং (CLAHE)",
    desc: "Removing metallic aluminum glare and sharpening embossed dosage characters",
  },
  {
    step: 3,
    title: "PaddleOCR PP-OCRv6 Multi-Orientation",
    titleBn: "ধাপ ৩/৬: PP-OCRv6 টেক্সট ও মাত্রা (mg) এক্সট্রাকশন",
    desc: "Running 0° + 90° CCW neural text detection and recognition across strip crops",
  },
  {
    step: 4,
    title: "RapidFuzz + SQLite 17,191 BD Medicine Index",
    titleBn: "ধাপ ৪/৬: RapidFuzz ও ১৭,১৯১টি ওষুধের ডাটাবেস ম্যাচিং",
    desc: "Spatial brand-anchor clustering and fuzzy entity resolution against medicines.db",
  },
  {
    step: 5,
    title: "Fine-Tuned Gemma 3 1B Bangla SLM (Q4_K_M)",
    titleBn: "ধাপ ৫/৬: Gemma 3 1B বাংলা সারসংক্ষেপ ও ডুপ্লিকেট যাচাই",
    desc: "Generating grounded Bangla summaries via local llama.cpp Gemma 3 1B & checking pairwise drug conflicts",
  },
  {
    step: 6,
    title: "Bangla Neural Voice Synthesis (edge-tts)",
    titleBn: "ধাপ ৬/৬: বাংলা নিউরাল ভয়েস (bn-BD-NabanitaNeural) তৈরি",
    desc: "Synthesizing spoken Bangla medication guidance and daily schedule audio",
  },
];

const EQUALIZER_HEIGHTS = [40, 85, 55, 100, 65, 90, 45, 75, 60, 95, 50, 80];

function formatAudioTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function splitBanglaLines(script: string): string[] {
  return (script || "")
    .replace(/\s*(সতর্কতা\s*:)/g, "\n$1")
    .split(/(?<=।)\s*|\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export default function PipelineInspectorPage() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(
    null
  );
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<ScanAnalysisResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [activeStageIdx, setActiveStageIdx] = useState<number>(0);
  const [liveStageStatuses, setLiveStageStatuses] = useState<
    Array<"pending" | "running" | "done">
  >(["pending", "pending", "pending", "pending", "pending", "pending"]);
  const [liveStageTimings, setLiveStageTimings] = useState<
    Array<number | null>
  >([null, null, null, null, null, null]);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [voiceLoadedForKey, setVoiceLoadedForKey] = useState<string | null>(
    null
  );
  const [voiceFailedForKey, setVoiceFailedForKey] = useState<string | null>(
    null
  );
  const [readyAudioBlobUrl, setReadyAudioBlobUrl] = useState<string | null>(
    null
  );
  const [readyAudioEl, setReadyAudioEl] = useState<HTMLAudioElement | null>(
    null
  );
  const [isPlayingVoice, setIsPlayingVoice] = useState<boolean>(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [audioCurrentTime, setAudioCurrentTime] = useState<number>(0);
  const [activeMedicine, setActiveMedicine] =
    useState<MedicineDatabaseRecord | null>(null);
  const [detectorMode, setDetectorMode] = useState<
    "hybrid_fusion" | "yolo_only" | "ocr_only"
  >("hybrid_fusion");
  const [yoloConf, setYoloConf] = useState<number>(0.22);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadedPreviewUrlRef = useRef<string | null>(null);

  const activeVoiceKey = analysis?.audioUrl
    ? `${analysis.scanId}:${analysis.audioUrl}`
    : null;
  const isVoiceLoading = Boolean(
    activeVoiceKey &&
      voiceLoadedForKey !== activeVoiceKey &&
      voiceFailedForKey !== activeVoiceKey
  );
  const isVoiceReady = Boolean(
    activeVoiceKey && voiceLoadedForKey === activeVoiceKey && readyAudioBlobUrl
  );

  // Lock background page scroll and allow Escape key to close clinical modal cleanly
  useEffect(() => {
    if (!activeMedicine) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setActiveMedicine(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeMedicine]);

  useEffect(() => {
    uploadedPreviewUrlRef.current = uploadedPreviewUrl;
  }, [uploadedPreviewUrl]);

  // Revoke preview URL & stop audio playback on unmount
  useEffect(() => {
    return () => {
      if (uploadedPreviewUrlRef.current?.startsWith("blob:")) {
        URL.revokeObjectURL(uploadedPreviewUrlRef.current);
      }
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const resetVoiceAudioState = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.onloadedmetadata = null;
      audioRef.current.ondurationchange = null;
      audioRef.current.ontimeupdate = null;
      audioRef.current.onplay = null;
      audioRef.current.onpause = null;
      audioRef.current.onended = null;
      audioRef.current.onerror = null;
      audioRef.current.src = "";
      audioRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setReadyAudioBlobUrl(null);
    setReadyAudioEl(null);
    setVoiceLoadedForKey(null);
    setVoiceFailedForKey(null);
    setAudioDuration(0);
    setAudioCurrentTime(0);
    setIsPlayingVoice(false);
  }, []);

  const handleAudioSeek = useCallback((sec: number) => {
    setAudioCurrentTime(sec);
  }, []);

  // Pre-fetch Bangla TTS audio asynchronously in background after scan results appear
  useEffect(() => {
    if (!activeVoiceKey || !analysis?.audioUrl) {
      return;
    }
    let cancelled = false;
    let createdObjUrl: string | null = null;
    const controller = new AbortController();
    const tVoice0 = Date.now();

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    fetchBanglaTtsAudioBlob(
      analysis.banglaVoiceScript || "",
      analysis.audioUrl,
      controller.signal
    )
      .then((blob) => {
        if (cancelled) return;
        const voiceElapsed = Math.max(12, Math.round(Date.now() - tVoice0));
        const objUrl = URL.createObjectURL(blob);
        createdObjUrl = objUrl;
        setReadyAudioBlobUrl(objUrl);

        const audio = new Audio(objUrl);
        audio.preload = "metadata";
        audioRef.current = audio;
        setReadyAudioEl(audio);

        const syncDuration = () => {
          if (!cancelled && Number.isFinite(audio.duration) && audio.duration > 0) {
            setAudioDuration(audio.duration);
          }
        };
        audio.onloadedmetadata = syncDuration;
        audio.ondurationchange = syncDuration;
        audio.ontimeupdate = () => {
          if (!cancelled) {
            setAudioCurrentTime(audio.currentTime || 0);
          }
        };
        audio.onplay = () => {
          if (!cancelled) setIsPlayingVoice(true);
        };
        audio.onpause = () => {
          if (!cancelled) setIsPlayingVoice(false);
        };
        audio.onended = () => {
          if (!cancelled) {
            setIsPlayingVoice(false);
            setAudioCurrentTime(
              Number.isFinite(audio.duration) ? audio.duration : 0
            );
          }
        };
        audio.onerror = () => {
          if (!cancelled) setIsPlayingVoice(false);
        };

        setVoiceLoadedForKey(activeVoiceKey);
        setLiveStageStatuses((prev) => {
          const next = [...prev];
          next[5] = "done";
          return next;
        });
        setLiveStageTimings((prev) => {
          const next = [...prev];
          next[5] = voiceElapsed;
          return next;
        });
        setAnalysis((prev) => {
          if (!prev || !prev.transparencyInspector) return prev;
          const updatedStages = prev.transparencyInspector.stages.map((st) =>
            st.id === "voice_synthesis" ? { ...st, durationMs: voiceElapsed } : st
          );
          const newTotal = updatedStages.reduce((acc, s) => acc + s.durationMs, 0);
          return {
            ...prev,
            inferenceTimeMs: newTotal,
            transparencyInspector: {
              ...prev.transparencyInspector,
              totalTurnaroundMs: newTotal,
              stages: updatedStages,
            },
          };
        });
      })
      .catch(() => {
        if (!cancelled) {
          setVoiceFailedForKey(activeVoiceKey);
          setLiveStageStatuses((prev) => {
            const next = [...prev];
            next[5] = "done";
            return next;
          });
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.onloadedmetadata = null;
        audioRef.current.ondurationchange = null;
        audioRef.current.ontimeupdate = null;
        audioRef.current.onplay = null;
        audioRef.current.onpause = null;
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.src = "";
        audioRef.current = null;
      }
      setReadyAudioEl(null);
      setIsPlayingVoice(false);
      if (createdObjUrl) {
        URL.revokeObjectURL(createdObjUrl);
      }
    };
  }, [activeVoiceKey, analysis?.audioUrl, analysis?.banglaVoiceScript]);

  const executeUploadedScan = useCallback(async (file: File) => {
    resetVoiceAudioState();
    setIsRunning(true);
    setActiveStageIdx(0);
    setLiveStageStatuses([
      "running",
      "pending",
      "pending",
      "pending",
      "pending",
      "pending",
    ]);
    setLiveStageTimings([null, null, null, null, null, null]);
    setElapsedMs(0);

    const t0 = Date.now();
    const stageTimer = setInterval(() => {
      setElapsedMs(Math.round(Date.now() - t0));
    }, 100);

    try {
      const { data, isLiveBackend } = await analyzeMedicineImageStream(
        file,
        (ev) => {
          if (ev.stageIndex < 0 || ev.stageIndex >= 5) return;
          setActiveStageIdx(ev.stageIndex);
          setLiveStageStatuses((prev) => {
            const next = [...prev];
            for (let i = 0; i < ev.stageIndex; i++) {
              next[i] = "done";
            }
            next[ev.stageIndex] = ev.status;
            return next;
          });
          if (ev.status === "done" && typeof ev.durationMs === "number") {
            setLiveStageTimings((prev) => {
              const next = [...prev];
              next[ev.stageIndex] = ev.durationMs ?? null;
              return next;
            });
          }
        },
        { detectorMode, yoloConf }
      );
      setReadyAudioBlobUrl(null);
      setReadyAudioEl(null);
      setAudioDuration(0);
      setAudioCurrentTime(0);
      setIsPlayingVoice(false);
      setLiveStageStatuses([
        "done",
        "done",
        "done",
        "done",
        "done",
        data.audioUrl ? "running" : "done",
      ]);
      setActiveStageIdx(5);
      setAnalysis(data);
      toast.success(
        isLiveBackend
          ? `Live 6-Stage Pipeline Completed (${data.inferenceTimeMs} ms)`
          : "Pipeline Scan Completed",
        {
          description: `${data.detectedStrips.length} medicine strip(s) verified with real telemetry`,
        }
      );
    } catch (err) {
      setLiveStageStatuses([
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
      ]);
      toast.error("Pipeline Scan Failed", {
        description:
          err instanceof Error
            ? err.message
            : "Could not analyze the uploaded image. Please try again.",
      });
    } finally {
      clearInterval(stageTimer);
      setIsRunning(false);
    }
  }, [detectorMode, yoloConf, resetVoiceAudioState]);

  const acceptFile = (file: File | undefined | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    if (uploadedPreviewUrl) {
      URL.revokeObjectURL(uploadedPreviewUrl);
    }
    resetVoiceAudioState();
    setUploadedFile(file);
    setUploadedPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
    setLiveStageStatuses([
      "pending",
      "pending",
      "pending",
      "pending",
      "pending",
      "pending",
    ]);
    setLiveStageTimings([null, null, null, null, null, null]);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    acceptFile(e.target.files?.[0]);
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isRunning) return;
    acceptFile(e.dataTransfer.files?.[0]);
  };

  const handleClearImage = () => {
    if (uploadedPreviewUrl) {
      URL.revokeObjectURL(uploadedPreviewUrl);
    }
    resetVoiceAudioState();
    setUploadedFile(null);
    setUploadedPreviewUrl(null);
    setAnalysis(null);
    setLiveStageStatuses([
      "pending",
      "pending",
      "pending",
      "pending",
      "pending",
      "pending",
    ]);
    setLiveStageTimings([null, null, null, null, null, null]);
  };

  const handleRunActiveImage = () => {
    if (!uploadedFile) {
      fileInputRef.current?.click();
      return;
    }
    executeUploadedScan(uploadedFile);
  };

  const toggleBanglaVoice = () => {
    if (!analysis || typeof window === "undefined") return;
    if (isPlayingVoice) {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      setIsPlayingVoice(false);
      return;
    }

    if (audioRef.current && readyAudioBlobUrl) {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      if (
        audioDuration > 0 &&
        audioRef.current.currentTime >= audioDuration - 0.1
      ) {
        audioRef.current.currentTime = 0;
        setAudioCurrentTime(0);
      }
      setIsPlayingVoice(true);
      audioRef.current.play().catch(() => setIsPlayingVoice(false));
      return;
    }

    if ("speechSynthesis" in window) {
      setIsPlayingVoice(true);
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(analysis.banglaVoiceScript);
      u.lang = "bn-BD";
      u.rate = 0.95;
      u.onend = () => setIsPlayingVoice(false);
      u.onerror = () => setIsPlayingVoice(false);
      window.speechSynthesis.speak(u);
    }
  };

  const audioProgressPct =
    audioDuration > 0
      ? Math.min(100, Math.round((audioCurrentTime / audioDuration) * 100))
      : 0;

  const handleOpenStripModal = (strip: BoundingBox) => {
    if (strip.databaseRecord) {
      setActiveMedicine({
        ...strip.databaseRecord,
        package_container:
          strip.strength ||
          strip.databaseRecord.package_container ||
          "Standard Dose",
      });
    }
  };

  const summaryLines = analysis
    ? splitBanglaLines(analysis.banglaVoiceScript)
    : [];

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 pt-8 sm:pt-10 pb-20 space-y-8">
        {/* Page Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-white/90 px-4 py-1 text-xs font-extrabold text-[#0ea86b] shadow-2xs border border-white">
            <Cpu className="size-3.5" />
            Live Multi-Stage Edge AI Pipeline &amp; Forensic Proof
          </span>
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-[#09353c]">
            AI Pipeline &amp; Transparency Inspector
          </h1>
          <p className="text-xs sm:text-sm text-[#46737a] leading-relaxed">
            Upload your own medicine strip photo just like on the scanner page,
            then run the live 6-stage pipeline to inspect per-stage millisecond
            timings, cropped &amp; CLAHE-rectified foil thumbnails, raw PP-OCRv6
            lines, RapidFuzz match scores, and Bangla SLM summaries.
          </p>
        </div>

        {/* Upload Workspace + Live 6-Stage Pipeline Tracker */}
        <div className="rounded-[2.25rem] bg-white/95 backdrop-blur-2xl p-6 sm:p-8 shadow-[0_24px_70px_rgba(11,85,99,0.09)] border border-white space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-teal-100 pb-5">
            <div>
              <h2 className="text-lg sm:text-xl font-extrabold text-[#09353c]">
                1. Upload Medicine Strip Photo
              </h2>
              <p className="text-xs text-[#46737a] mt-1">
                Nothing runs automatically: upload any medicine image and click
                Run whenever you want to inspect the full AI pipeline.
              </p>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                disabled={isRunning}
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-xs font-bold border bg-emerald-50 text-[#0b5563] border-[#0ea86b]/40 hover:bg-emerald-100 transition cursor-pointer disabled:opacity-50"
              >
                <Upload className="size-3.5" />
                <span>{uploadedFile ? "Change Image" : "Upload Image"}</span>
              </button>
              {uploadedFile && (
                <button
                  type="button"
                  disabled={isRunning}
                  onClick={handleClearImage}
                  className="inline-flex items-center gap-1.5 rounded-full px-3.5 py-2 text-xs font-bold border bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100 transition cursor-pointer disabled:opacity-50"
                >
                  <Trash2 className="size-3.5" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Left Upload Dropzone / Preview + Right Live 6-Stage Progress Tracker */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            {/* Upload Dropzone or Uploaded Image Preview */}
            <div className="lg:col-span-5 space-y-3">
              {uploadedPreviewUrl ? (
                <div
                  onClick={() => !isRunning && fileInputRef.current?.click()}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  title="Click or drop to replace image"
                  className={`relative rounded-3xl overflow-hidden border-2 bg-[#09353c]/5 aspect-[4/3] shadow-sm cursor-pointer transition ${
                    isDragging
                      ? "border-[#0ea86b] ring-4 ring-[#0ea86b]/20"
                      : "border-teal-100 hover:border-[#0ea86b]/60"
                  }`}
                >
                  <div className="relative w-full h-full flex items-center justify-center p-2">
                    <div className="relative inline-block max-w-full max-h-full">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={uploadedPreviewUrl}
                        alt={uploadedFile?.name || "Uploaded medicine strips"}
                        className="block max-w-full max-h-[300px] w-auto h-auto rounded-2xl"
                      />
                      {showBoxes &&
                        analysis?.detectedStrips.map((strip, idx) => {
                          const isYolo = Boolean(
                            strip.detectionSource?.startsWith("yolo26")
                          );
                          return (
                            <div
                              key={strip.id}
                              style={{
                                left: `${strip.x}%`,
                                top: `${strip.y}%`,
                                width: `${strip.width}%`,
                                height: `${strip.height}%`,
                              }}
                              className={`pointer-events-none absolute rounded-lg border-2 ${
                                isYolo
                                  ? "border-[#0ea86b] bg-[#0ea86b]/10"
                                  : "border-dashed border-amber-400 bg-amber-400/10"
                              }`}
                            >
                              <span
                                className={`absolute -top-5 left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[9px] font-bold text-white shadow ${
                                  isYolo ? "bg-[#0ea86b]" : "bg-amber-600"
                                }`}
                              >
                                #{idx + 1} {strip.brandName}
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                  <div className="absolute bottom-3 left-3 right-3 rounded-2xl bg-[#09353c]/85 backdrop-blur-md text-white px-3.5 py-2 text-xs flex items-center justify-between">
                    <span className="font-bold truncate">
                      {uploadedFile?.name}
                    </span>
                    <span className="font-mono text-[11px] text-emerald-300 shrink-0 ml-2">
                      {analysis
                        ? `${analysis.detectedStrips.length} Strips`
                        : "Ready"}
                    </span>
                  </div>
                </div>
              ) : (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className={`rounded-3xl border-2 border-dashed aspect-[4/3] flex flex-col items-center justify-center p-6 text-center cursor-pointer transition ${
                    isDragging
                      ? "border-[#0ea86b] bg-emerald-50/70"
                      : "border-teal-200 bg-[#f2f9f8]/80 hover:border-[#0ea86b] hover:bg-emerald-50/40"
                  }`}
                >
                  <div className="flex items-center justify-center size-14 rounded-2xl bg-white shadow-sm border border-teal-100 text-[#0ea86b] mb-3">
                    <ImagePlus className="size-7" />
                  </div>
                  <p className="text-sm font-extrabold text-[#09353c]">
                    ওষুধের পাতার ছবি আপলোড করুন
                  </p>
                  <p className="text-xs text-[#46737a] mt-1 max-w-xs">
                    Click to browse or drag &amp; drop a tabletop photo of 1–6
                    medicine blister strips (JPG, PNG, WEBP)
                  </p>
                </div>
              )}

              {/* Compact Live Detector Controls (Mode & Confidence Override) */}
              <div className="rounded-2xl border border-teal-100 bg-[#f2f9f8]/90 px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-[#09353c]">Detector:</span>
                  {(["hybrid_fusion", "yolo_only", "ocr_only"] as const).map(
                    (m) => (
                      <button
                        key={m}
                        type="button"
                        disabled={isRunning}
                        onClick={() => setDetectorMode(m)}
                        className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold transition cursor-pointer ${
                          detectorMode === m
                            ? "bg-[#09353c] text-white"
                            : "bg-white text-[#46737a] border border-teal-100 hover:border-[#0ea86b]"
                        }`}
                      >
                        {m === "hybrid_fusion"
                          ? "Hybrid"
                          : m === "yolo_only"
                          ? "YOLO Only"
                          : "OCR Only"}
                      </button>
                    )
                  )}
                </div>

                <div className="flex items-center gap-2.5">
                  <label className="flex items-center gap-1.5 text-[11px] font-mono text-[#0b5563]">
                    <span>conf≥{yoloConf.toFixed(2)}</span>
                    <input
                      type="range"
                      min={0.1}
                      max={0.85}
                      step={0.05}
                      value={yoloConf}
                      disabled={isRunning}
                      onChange={(e) => setYoloConf(Number(e.target.value))}
                      className="w-16 accent-[#0ea86b]"
                    />
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowBoxes((v) => !v)}
                    className="rounded-full bg-white px-2 py-0.5 text-[10px] font-bold text-[#0b5563] border border-teal-100 hover:border-[#0ea86b] cursor-pointer"
                  >
                    {showBoxes ? "Hide Boxes" : "Show Boxes"}
                  </button>
                </div>
              </div>

              <button
                type="button"
                disabled={isRunning || !uploadedFile}
                onClick={handleRunActiveImage}
                className="w-full inline-flex items-center justify-center gap-2 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs sm:text-sm font-bold py-3.5 shadow-md shadow-[#0ea86b]/25 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    <span>
                      Running Stage {activeStageIdx + 1}/6 ({elapsedMs} ms)...
                    </span>
                  </>
                ) : (
                  <>
                    <Play className="size-4 fill-current" />
                    <span>
                      {uploadedFile
                        ? "Run Live 6-Stage Pipeline on Uploaded Image"
                        : "Upload an Image First to Run Pipeline"}
                    </span>
                  </>
                )}
              </button>
            </div>

            {/* Live 6-Stage Pipeline Step-by-Step Progress Checklist */}
            <div className="lg:col-span-7 space-y-2.5">
              <div className="flex items-center justify-between px-1">
                <span className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                  Live 6-Stage Pipeline Execution Status
                </span>
                <span className="font-mono text-xs font-bold text-[#0ea86b]">
                  {isRunning
                    ? `Executing Stage ${activeStageIdx + 1}/6... (${elapsedMs} ms)`
                    : isVoiceLoading
                    ? "Synthesizing Bangla Voice (Stage 6/6)..."
                    : analysis
                    ? `Completed in ${analysis.inferenceTimeMs} ms`
                    : "Waiting for Uploaded Image"}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {PIPELINE_STAGE_STEPS.map((st, idx) => {
                  const isVoiceStage = idx === 5;
                  const liveStatus = liveStageStatuses[idx];
                  const inspectorTiming =
                    analysis?.transparencyInspector?.stages[idx]?.durationMs;
                  const stageTiming =
                    liveStageTimings[idx] !== null
                      ? liveStageTimings[idx]
                      : inspectorTiming;

                  const isCurrent = isRunning
                    ? liveStatus === "running"
                    : isVoiceStage && Boolean(analysis) && isVoiceLoading;

                  const isDone = isRunning
                    ? liveStatus === "done"
                    : Boolean(analysis) && !(isVoiceStage && isVoiceLoading);

                  return (
                    <div
                      key={st.step}
                      className={`rounded-2xl p-3.5 border transition-all ${
                        isCurrent
                          ? "bg-emerald-50/90 border-[#0ea86b] shadow-sm"
                          : isDone
                          ? "bg-[#f2f9f8] border-teal-100"
                          : "bg-white/80 border-teal-100/70"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          {isCurrent ? (
                            <Loader2 className="size-4 text-[#0ea86b] animate-spin shrink-0" />
                          ) : isDone ? (
                            <span className="flex items-center justify-center size-5 rounded-full bg-[#0ea86b] text-white shrink-0">
                              <Check className="size-3 stroke-[3]" />
                            </span>
                          ) : (
                            <span className="flex items-center justify-center size-5 rounded-full bg-teal-100 text-[#46737a] text-[10px] font-bold shrink-0">
                              {st.step}
                            </span>
                          )}
                          <span className="text-xs font-extrabold text-[#09353c]">
                            {st.titleBn}
                          </span>
                        </div>
                        {isDone &&
                          stageTiming !== undefined &&
                          stageTiming !== null && (
                            <span className="font-mono text-[11px] font-extrabold text-[#0ea86b] bg-white px-2 py-0.5 rounded-md border border-teal-100 shrink-0">
                              {stageTiming} ms
                            </span>
                          )}
                      </div>
                      <p className="text-[11px] text-[#46737a] mt-1 pl-7 leading-snug">
                        {st.desc}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Bangla Line-by-Line Summary & Detected Medicines Cards */}
        {analysis && (
          <div className="rounded-[2.25rem] bg-white/95 backdrop-blur-2xl p-6 sm:p-8 shadow-[0_24px_70px_rgba(11,85,99,0.09)] border border-white space-y-5">
            <div className="rounded-2xl bg-[#f2f9f8] p-5 border border-teal-100 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-teal-200/60 pb-3.5">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                    বাংলা ভয়েস গাইড ও সারসংক্ষেপ (Bangla Summary)
                  </p>
                  {isVoiceLoading ? (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 px-2.5 py-0.5 text-[10px] font-extrabold">
                      <Loader2 className="size-3 animate-spin text-amber-600" />
                      ভয়েস অডিও তৈরি হচ্ছে...
                    </span>
                  ) : isVoiceReady ? (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200 px-2.5 py-0.5 text-[10px] font-extrabold">
                      <Check className="size-3 stroke-[3]" />
                      ভয়েস প্রস্তুত
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-teal-50 text-[#46737a] border border-teal-200 px-2.5 py-0.5 text-[10px] font-extrabold">
                      ব্রাউজার ভয়েস
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={toggleBanglaVoice}
                  className="self-start sm:self-center shrink-0 inline-flex items-center gap-2 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] px-5 py-2 text-xs font-bold text-white shadow-sm transition cursor-pointer"
                >
                  {isPlayingVoice ? (
                    <>
                      <Pause className="size-3.5 fill-current" />
                      <span>Pause</span>
                    </>
                  ) : isVoiceLoading ? (
                    <>
                      <Loader2 className="size-3.5 animate-spin" />
                      <span>ভয়েস তৈরি হচ্ছে (তাৎক্ষণিক শুনুন)</span>
                    </>
                  ) : (
                    <>
                      <Volume2 className="size-4" />
                      <span>বাংলায় শুনুন</span>
                    </>
                  )}
                </button>
              </div>

              {/* Dedicated Interactive Bangla Voice Player (WaveSurfer.js + Duration & Progress) */}
              <div className="rounded-2xl bg-gradient-to-r from-[#09353c] via-[#0b5563] to-[#0d6e6e] p-4 sm:p-5 text-white shadow-md border border-teal-700/40 space-y-3.5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3.5">
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={toggleBanglaVoice}
                      disabled={isVoiceLoading}
                      className={`flex items-center justify-center size-11 rounded-2xl shadow-md transition cursor-pointer shrink-0 ${
                        isVoiceLoading
                          ? "bg-white/15 text-emerald-300 cursor-wait"
                          : isPlayingVoice
                          ? "bg-white text-[#09353c] hover:bg-emerald-50"
                          : "bg-[#0ea86b] text-white hover:bg-emerald-400"
                      }`}
                      aria-label={
                        isPlayingVoice
                          ? "Pause Bangla Audio"
                          : "Play Bangla Audio"
                      }
                    >
                      {isVoiceLoading ? (
                        <Loader2 className="size-5 animate-spin" />
                      ) : isPlayingVoice ? (
                        <Pause className="size-5 fill-current" />
                      ) : (
                        <Play className="size-5 fill-current ml-0.5" />
                      )}
                    </button>

                    <div className="space-y-0.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs sm:text-sm font-extrabold tracking-wide flex items-center gap-1.5">
                          <Volume2 className="size-4 text-emerald-300" />
                          বাংলা ভয়েস প্লেয়ার (Bangla Voice Player)
                        </span>
                        {isVoiceLoading ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/20 text-amber-200 border border-amber-300/30 px-2.5 py-0.5 text-[10px] font-bold">
                            <Loader2 className="size-3 animate-spin" />
                            অডিও তৈরি হচ্ছে...
                          </span>
                        ) : isVoiceReady ? (
                          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-400/20 text-emerald-200 border border-emerald-300/30 px-2.5 py-0.5 text-[10px] font-bold">
                            <Check className="size-3 stroke-[3]" />
                            {isPlayingVoice
                              ? "চলছে (Playing)"
                              : "প্রস্তুত (Ready)"}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 rounded-full bg-white/15 text-teal-100 border border-white/20 px-2.5 py-0.5 text-[10px] font-bold">
                            ব্রাউজার ভয়েস (Fallback)
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-teal-100/85">
                        {isVoiceLoading
                          ? "ধাপ ৬/৬: বাংলা নিউরাল ভয়েস অডিও ব্যাকগ্রাউন্ডে প্রস্তুত হচ্ছে..."
                          : isVoiceReady
                          ? `মোট দৈর্ঘ্য: ${formatAudioTime(audioDuration)} • সম্পন্ন: ${audioProgressPct}%`
                          : "নিউরাল অডিও পাওয়া যায়নি: ব্রাউজার ভয়েস দিয়ে শুনুন"}
                      </p>
                    </div>
                  </div>

                  {/* Animated Soundwave Equalizer Bars */}
                  <div className="flex items-end justify-center sm:justify-end gap-1 h-9 px-3 py-1.5 rounded-2xl bg-white/10 border border-white/10">
                    {EQUALIZER_HEIGHTS.map((h, i) => (
                      <span
                        key={i}
                        style={{
                          height: isPlayingVoice
                            ? `${h}%`
                            : isVoiceLoading
                            ? `${Math.max(25, Math.round(h * 0.5))}%`
                            : "22%",
                          animationDelay: `${i * 90}ms`,
                        }}
                        className={`w-1 rounded-full transition-all duration-300 ${
                          isPlayingVoice
                            ? "bg-emerald-300 animate-pulse"
                            : isVoiceLoading
                            ? "bg-amber-300/70 animate-pulse"
                            : "bg-teal-200/50"
                        }`}
                      />
                    ))}
                  </div>
                </div>

                {/* WaveSurfer.js Interactive Waveform & Progress Row */}
                <div className="space-y-1.5">
                  {isVoiceLoading || readyAudioBlobUrl ? (
                    <BanglaVoiceWaveform
                      audioUrl={readyAudioBlobUrl}
                      audioElement={readyAudioEl}
                      isLoading={isVoiceLoading}
                      onSeek={handleAudioSeek}
                    />
                  ) : (
                    <div className="h-11 w-full rounded-xl bg-white/10 border border-white/15 flex items-center justify-center px-3 text-[11px] font-semibold text-teal-100/90">
                      ব্রাউজার ভয়েস প্রস্তুত: শুনতে Play অথবা &ldquo;বাংলায় শুনুন&rdquo; ক্লিক করুন
                    </div>
                  )}
                  <div className="flex items-center justify-between text-[11px] font-mono text-teal-100/90">
                    <span>{formatAudioTime(audioCurrentTime)}</span>
                    <span className="font-bold text-emerald-300">
                      {isVoiceLoading
                        ? "Generating MP3..."
                        : isVoiceReady
                        ? `${audioProgressPct}% Done`
                        : isPlayingVoice
                        ? "Speaking (Browser Voice)..."
                        : "Browser Voice Ready"}
                    </span>
                    <span>
                      {audioDuration > 0
                        ? formatAudioTime(audioDuration)
                        : "--:--"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Clean Line-by-Line Bangla Summary (No Paragraph Block) */}
              <ul className="space-y-2">
                {summaryLines.map((line, i) => {
                  const isFirstLine = i === 0;
                  const isStripLine =
                    line.startsWith("ওষুধ ") || line.startsWith("স্ট্রিপ");
                  const isWarningLine = line.startsWith("সতর্কতা");
                  return (
                    <li
                      key={`${i}-${line.slice(0, 16)}`}
                      className={`flex items-start gap-2.5 text-xs sm:text-sm leading-relaxed ${
                        isFirstLine
                          ? "font-extrabold text-[#0b5563]"
                          : isWarningLine
                          ? "font-semibold text-amber-900 bg-amber-50/90 rounded-xl px-3.5 py-2 border border-amber-200"
                          : isStripLine
                          ? "font-semibold text-[#09353c] bg-white/85 rounded-xl px-3.5 py-2 border border-teal-100/80"
                          : "font-bold text-[#0ea86b] pt-1"
                      }`}
                    >
                      {isStripLine && (
                        <span className="mt-1.5 size-2 rounded-full bg-[#0ea86b] shrink-0" />
                      )}
                      <span>{line}</span>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {analysis.detectedStrips.map((strip) => (
                <div
                  key={strip.id}
                  onClick={() => handleOpenStripModal(strip)}
                  className="group rounded-2xl bg-white p-4 border border-teal-100 shadow-2xs hover:shadow-md hover:-translate-y-0.5 transition cursor-pointer flex flex-col justify-between space-y-3"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] font-bold text-[#0b5563] bg-[#f2f9f8] px-2.5 py-0.5 rounded-full border border-teal-100">
                        {strip.dosageForm || "Tablet"}
                      </span>
                      <span className="rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold">
                        {strip.strength}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <h3 className="font-extrabold text-base text-[#09353c] group-hover:text-[#0ea86b] transition">
                        {strip.brandName} ({strip.strength})
                      </h3>
                      <ArrowUpRight className="size-4 text-[#0d9488]" />
                    </div>
                    <p className="text-xs font-bold text-[#0d9488]">
                      {strip.genericName}
                    </p>
                    <p className="text-[11px] text-[#09353c] bg-[#f2f9f8] p-2.5 rounded-xl border border-teal-50 leading-relaxed whitespace-pre-line">
                      {strip.databaseRecord?.beginner_summary ||
                        strip.dosageBangla}
                    </p>
                  </div>
                  <div className="pt-2 border-t border-teal-50 flex items-center justify-between text-[11px] font-bold text-[#0ea86b]">
                    <span className="inline-flex items-center gap-1">
                      <Database className="size-3.5" />
                      বিস্তারিত ডাটাবেস তথ্য দেখুন
                    </span>
                    <span>Click →</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Full AI Transparency Inspector (appears once you run a scan) */}
        {(analysis || isRunning) && (
          <AITransparencyInspector
            analysis={analysis}
            isRunning={isRunning}
            activeStageIndex={activeStageIdx}
            onRerun={handleRunActiveImage}
          />
        )}
      </main>

      {/* Clinical Detail Modal */}
      {activeMedicine && (
        <div
          onClick={() => setActiveMedicine(null)}
          data-lenis-prevent="true"
          className="fixed inset-0 z-50 bg-[#09353c]/55 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            onWheel={(e) => e.stopPropagation()}
            onTouchMove={(e) => e.stopPropagation()}
            data-lenis-prevent="true"
            className="relative w-full max-w-2xl max-h-[86vh] overflow-y-auto overscroll-contain rounded-[2.25rem] bg-white p-6 sm:p-8 shadow-2xl border border-teal-100 space-y-6"
          >
            <div className="flex items-start justify-between gap-4 border-b border-teal-100 pb-5">
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 text-[#0ea86b] px-3 py-1 text-xs font-bold border border-emerald-200">
                    <Pill className="size-3.5" />
                    {activeMedicine.dosage_form || "Medicine"}
                  </span>
                  {activeMedicine.drug_class && (
                    <span className="rounded-full bg-[#f2f9f8] text-[#0b5563] border border-teal-100 px-3 py-1 text-xs font-semibold">
                      {activeMedicine.drug_class}
                    </span>
                  )}
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-[#09353c] pt-1">
                  {activeMedicine.brand_name}
                </h2>
                <p className="text-sm font-bold text-[#0d9488]">
                  Generic Compound: {activeMedicine.generic}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setActiveMedicine(null)}
                className="flex items-center justify-center size-9 rounded-full bg-[#f2f9f8] hover:bg-teal-100 text-[#09353c] transition cursor-pointer shrink-0"
              >
                <X className="size-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs">
              <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-1">
                <span className="font-bold text-[#46737a] flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
                  <Building2 className="size-3.5 text-[#0ea86b]" />
                  Manufacturer
                </span>
                <p className="font-extrabold text-sm text-[#09353c]">
                  {activeMedicine.manufacturer || "N/A"}
                </p>
              </div>
              <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100/80 space-y-1">
                <span className="font-bold text-[#46737a] flex items-center gap-1.5 uppercase tracking-wider text-[10px]">
                  <Stethoscope className="size-3.5 text-[#0ea86b]" />
                  Detected Strength &amp; Pack Info
                </span>
                <p className="font-extrabold text-sm text-[#09353c]">
                  {activeMedicine.package_container || "Standard Pack"}
                </p>
              </div>
            </div>

            {activeMedicine.beginner_summary && (
              <div className="rounded-3xl bg-gradient-to-br from-[#f2f9f8] via-emerald-50/60 to-[#f2f9f8] p-5 border-2 border-[#0ea86b]/30 shadow-xs space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                    <Sparkles className="size-4 text-[#0ea86b]" />
                    সহজ বাংলা সারসংক্ষেপ (Bangla Summary)
                  </p>
                  <span className="rounded-full bg-white px-2.5 py-0.5 text-[10px] font-bold text-[#0b5563] border border-teal-100">
                    Gemma 3 1B SLM
                  </span>
                </div>
                <p className="text-xs sm:text-sm font-medium text-[#09353c] leading-relaxed whitespace-pre-line">
                  {activeMedicine.beginner_summary}
                </p>
              </div>
            )}

            {activeMedicine.indication && (
              <div className="rounded-2xl bg-[#f2f9f8]/60 p-4 border border-teal-100 space-y-1.5">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563] flex items-center gap-1.5">
                  <Activity className="size-3.5 text-[#0ea86b]" />
                  Primary Indication
                </p>
                <p className="text-xs sm:text-sm font-semibold text-[#09353c] leading-relaxed">
                  {activeMedicine.indication}
                </p>
              </div>
            )}

            {activeMedicine.dosage_description && (
              <div className="rounded-3xl bg-white p-5 border border-teal-200/80 shadow-2xs space-y-2.5">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                  <Clock className="size-4 text-[#0ea86b]" />
                  Dosage &amp; Administration
                </p>
                <p className="text-xs sm:text-sm text-[#09353c] leading-relaxed">
                  {activeMedicine.dosage_description}
                </p>
              </div>
            )}

            {activeMedicine.interaction_description && (
              <div className="rounded-3xl bg-amber-50/70 p-5 border border-amber-200/90 space-y-2">
                <p className="text-xs font-extrabold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                  <AlertTriangle className="size-4 text-amber-600" />
                  Drug-Drug Interactions
                </p>
                <p className="text-xs sm:text-sm text-amber-950/90 leading-relaxed">
                  {activeMedicine.interaction_description}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      <Footer />
    </div>
  );
}
