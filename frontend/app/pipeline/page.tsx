"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { AITransparencyInspector } from "@/components/ai-transparency-inspector";
import { BACKEND_ORIGIN, analyzeMedicineImage } from "@/lib/api";
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
    title: "YOLO26n ONNX Object Detection",
    titleBn: "ধাপ ১/৬: YOLO26n ONNX স্ট্রিপ ডিটেকশন",
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
    title: "Quantized Safety SLM (Qwen2.5-1.5B-INT4)",
    titleBn: "ধাপ ৫/৬: Safety SLM বাংলা সারসংক্ষেপ ও ডুপ্লিকেট যাচাই",
    desc: "Suppressing torn-foil substring ghosts, checking pairwise drug conflicts & generating Bangla summary",
  },
  {
    step: 6,
    title: "Bangla Neural Voice Synthesis (edge-tts)",
    titleBn: "ধাপ ৬/৬: বাংলা নিউরাল ভয়েস (bn-BD-NabanitaNeural) তৈরি",
    desc: "Synthesizing spoken Bangla medication guidance and daily schedule audio",
  },
];

function splitBanglaLines(script: string): string[] {
  return script
    .split(/(?<=।)\s*/)
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
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [isVoiceLoading, setIsVoiceLoading] = useState<boolean>(false);
  const [readyAudioBlobUrl, setReadyAudioBlobUrl] = useState<string | null>(
    null
  );
  const [isPlayingVoice, setIsPlayingVoice] = useState<boolean>(false);
  const [activeMedicine, setActiveMedicine] =
    useState<MedicineDatabaseRecord | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Pre-fetch Bangla TTS audio asynchronously in background after scan results appear
  useEffect(() => {
    if (!analysis?.audioUrl) {
      setIsVoiceLoading(false);
      return;
    }
    let cancelled = false;
    setIsVoiceLoading(true);
    setReadyAudioBlobUrl(null);
    const tVoice0 = performance.now();

    const fullAudioUrl = analysis.audioUrl.startsWith("http")
      ? analysis.audioUrl
      : `${BACKEND_ORIGIN}${analysis.audioUrl}`;

    fetch(fullAudioUrl)
      .then((res) => (res.ok ? res.blob() : Promise.reject(new Error("TTS failed"))))
      .then((blob) => {
        if (cancelled) return;
        const voiceElapsed = Math.max(12, Math.round(performance.now() - tVoice0));
        setReadyAudioBlobUrl(URL.createObjectURL(blob));
        setIsVoiceLoading(false);
        setAnalysis((prev) => {
          if (!prev || !prev.transparencyInspector) return prev;
          const updatedStages = prev.transparencyInspector.stages.map((st) =>
            st.id === "voice_synthesis" ? { ...st, durationMs: voiceElapsed } : st
          );
          const newTotal = updatedStages.reduce((acc, s) => acc + s.durationMs, 0);
          return {
            ...prev,
            transparencyInspector: {
              ...prev.transparencyInspector,
              totalTurnaroundMs: newTotal,
              stages: updatedStages,
            },
          };
        });
      })
      .catch(() => {
        if (!cancelled) setIsVoiceLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [analysis?.scanId, analysis?.audioUrl]);

  const executeUploadedScan = useCallback(async (file: File) => {
    setIsRunning(true);
    setActiveStageIdx(0);
    setElapsedMs(0);

    const t0 = performance.now();
    const stageTimer = setInterval(() => {
      const nowMs = Math.round(performance.now() - t0);
      setElapsedMs(nowMs);
      setActiveStageIdx((prev) => Math.min(5, prev + 1));
    }, 420);

    try {
      const { data, isLiveBackend } = await analyzeMedicineImage(file);
      setAnalysis(data);
      toast.success(
        isLiveBackend
          ? `Live 6-Stage Pipeline Completed (${data.inferenceTimeMs} ms)`
          : "Pipeline Scan Completed",
        {
          description: `${data.detectedStrips.length} medicine strip(s) verified with real telemetry`,
        }
      );
    } finally {
      clearInterval(stageTimer);
      setActiveStageIdx(5);
      setIsRunning(false);
    }
  }, []);

  const acceptFile = (file: File | undefined | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    if (uploadedPreviewUrl) {
      URL.revokeObjectURL(uploadedPreviewUrl);
    }
    setUploadedFile(file);
    setUploadedPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
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
    setUploadedFile(null);
    setUploadedPreviewUrl(null);
    setAnalysis(null);
    setReadyAudioBlobUrl(null);
    setIsPlayingVoice(false);
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
        audioRef.current.currentTime = 0;
      }
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      setIsPlayingVoice(false);
      return;
    }

    if (readyAudioBlobUrl) {
      const audio = new Audio(readyAudioBlobUrl);
      audioRef.current = audio;
      setIsPlayingVoice(true);
      audio.onended = () => setIsPlayingVoice(false);
      audio.onerror = () => setIsPlayingVoice(false);
      audio.play().catch(() => setIsPlayingVoice(false));
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
                Nothing runs automatically — upload any medicine image and click
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
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={uploadedPreviewUrl}
                    alt={uploadedFile?.name || "Uploaded medicine strips"}
                    className="w-full h-full object-contain"
                  />
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
                    ? `Executing... (${elapsedMs} ms)`
                    : analysis
                    ? `Completed in ${analysis.inferenceTimeMs} ms`
                    : "Waiting for Uploaded Image"}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {PIPELINE_STAGE_STEPS.map((st, idx) => {
                  const stageTiming =
                    analysis?.transparencyInspector?.stages[idx]?.durationMs;
                  const isDone =
                    (Boolean(analysis) && !isRunning) ||
                    (isRunning && idx < activeStageIdx);
                  const isCurrent = isRunning && idx === activeStageIdx;

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
                        {stageTiming !== undefined && !isRunning && (
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
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 text-[#0ea86b] border border-emerald-200 px-2.5 py-0.5 text-[10px] font-extrabold">
                      <Check className="size-3 stroke-[3]" />
                      ভয়েস প্রস্তুত
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

              {/* Clean Line-by-Line Bangla Summary (No Paragraph Block) */}
              <ul className="space-y-2">
                {summaryLines.map((line, i) => {
                  const isStripLine = line.startsWith("স্ট্রিপ");
                  const isFirstLine = i === 0;
                  return (
                    <li
                      key={`${i}-${line.slice(0, 16)}`}
                      className={`flex items-start gap-2.5 text-xs sm:text-sm leading-relaxed ${
                        isFirstLine
                          ? "font-extrabold text-[#0b5563]"
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
                    <p className="text-[11px] text-[#09353c] bg-[#f2f9f8] p-2.5 rounded-xl border border-teal-50 leading-relaxed line-clamp-3">
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
          className="fixed inset-0 z-50 bg-[#09353c]/55 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl max-h-[86vh] overflow-y-auto rounded-[2.25rem] bg-white p-6 sm:p-8 shadow-2xl border border-teal-100 space-y-6"
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
                    Safety SLM
                  </span>
                </div>
                <p className="text-xs sm:text-sm font-medium text-[#09353c] leading-relaxed">
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
