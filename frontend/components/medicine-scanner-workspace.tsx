"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import Link from "next/link";
import Webcam from "react-webcam";
import { useDropzone } from "react-dropzone";
import {
  CloudUpload,
  Check,
  ChevronRight,
  Camera,
  Volume2,
  Pause,
  Play,
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Pill,
  Loader2,
  RotateCcw,
  X,
  Building2,
  Stethoscope,
  Clock,
  Activity,
  Database,
  ArrowUpRight,
  Cpu,
  Sparkles,
  BookOpen,
} from "lucide-react";
import { PRESET_SCENARIOS } from "@/lib/mock-data";
import {
  BoundingBox,
  MedicineDatabaseRecord,
  ScanAnalysisResult,
} from "@/lib/types";
import {
  analyzeMedicineImageStream,
  fetchBanglaTtsAudioBlob,
} from "@/lib/api";
import { BanglaVoiceWaveform } from "@/components/bangla-voice-waveform";
import { toast } from "sonner";

const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;

function splitBanglaSummaryLines(script: string): string[] {
  return (script || "")
    .replace(/\s*(সতর্কতা\s*:)/g, "\n$1")
    .split(/(?<=।)\s*|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

async function resizeImageForUpload(
  input: File | Blob,
  maxDim = 1920
): Promise<File | Blob> {
  if (typeof window === "undefined" || input.size < 400_000) return input;
  try {
    const bitmap = await createImageBitmap(input);
    const { width, height } = bitmap;
    if (Math.max(width, height) <= maxDim) {
      bitmap.close();
      return input;
    }
    const scale = maxDim / Math.max(width, height);
    const targetW = Math.max(1, Math.round(width * scale));
    const targetH = Math.max(1, Math.round(height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return input;
    }
    ctx.drawImage(bitmap, 0, 0, targetW, targetH);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.88)
    );
    if (!blob) return input;
    const name = input instanceof File ? input.name.replace(/\.[^.]+$/, ".jpg") : "upload.jpg";
    return new File([blob], name, { type: "image/jpeg" });
  } catch {
    return input;
  }
}

const LIVE_SCAN_STAGES = [
  {
    id: 1,
    labelBn: "ধাপ ১/৫: YOLO26s ONNX স্ট্রিপ ডিটেকশন চলছে...",
    labelEn: "Locating blister pack bounding boxes via YOLO26s ONNX",
    pct: 20,
  },
  {
    id: 2,
    labelBn: "ধাপ ২/৫: ফয়েল গ্লেয়ার রিমুভাল ও শার্পেনিং (CLAHE) চলছে...",
    labelEn: "Removing aluminum foil reflections & sharpening text",
    pct: 40,
  },
  {
    id: 3,
    labelBn: "ধাপ ৩/৫: PP-OCRv6 টেক্সট ও মাত্রা (mg) এক্সট্রাকশন চলছে...",
    labelEn: "Extracting brand names & strengths (0° + 90° CCW)",
    pct: 65,
  },
  {
    id: 4,
    labelBn: "ধাপ ৪/৫: RapidFuzz ও ১৭,১৯১টি ওষুধের ডাটাবেস ম্যাচিং চলছে...",
    labelEn: "Resolving brand & generic compound against medicines.db",
    pct: 85,
  },
  {
    id: 5,
    labelBn: "ধাপ ৫/৫: Gemma 3 1B বাংলা সারসংক্ষেপ ও ডুপ্লিকেট যাচাই চলছে...",
    labelEn: "Generating Bangla summary via Gemma 3 1B SLM & evaluating pairwise conflict rules",
    pct: 96,
  },
];

function formatAudioTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function parseStructuredClinicalText(raw: string): Array<{
  label: string | null;
  content: string;
}> {
  const cleaned = raw.replace(/\r\n/g, "\n").trim();
  if (!cleaned) return [];

  const withLineBreaks = cleaned.replace(
    /([.।!?])\s+([A-Z0-9\u0980-\u09FF][^:.।!?]{1,65}?)\s*:\s+/g,
    "$1\n$2: "
  );

  const rawLines = withLineBreaks
    .split(/\n+/)
    .map((l) => l.trim())
    .filter(Boolean);

  const items: Array<{ label: string | null; content: string }> = [];

  for (const line of rawLines) {
    const colonMatch = line.match(
      /^([A-Z0-9\u0980-\u09FF][^:]{1,68}?)\s*:\s+(.+)$/
    );
    if (colonMatch) {
      items.push({
        label: colonMatch[1].trim(),
        content: colonMatch[2].trim(),
      });
    } else {
      items.push({
        label: null,
        content: line,
      });
    }
  }

  return items;
}

export function MedicineScannerWorkspace() {
  const [step, setStep] = useState<"upload" | "summary">("upload");
  const [captureMode, setCaptureMode] = useState<"file" | "camera">("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStageIdx, setScanStageIdx] = useState(0);
  const [scanStageStatuses, setScanStageStatuses] = useState<
    Array<"pending" | "running" | "done">
  >(["pending", "pending", "pending", "pending", "pending"]);
  const [scanStageDurations, setScanStageDurations] = useState<
    Array<number | null>
  >([null, null, null, null, null]);
  const [scanElapsedMs, setScanElapsedMs] = useState(0);
  const [analysis, setAnalysis] = useState<ScanAnalysisResult>(
    PRESET_SCENARIOS[0].result
  );

  // Dedicated Bangla Audio Player state (length, current time, % done, playing, loading)
  const [voiceLoadedForKey, setVoiceLoadedForKey] = useState<string | null>(null);
  const [voiceFailedForKey, setVoiceFailedForKey] = useState<string | null>(null);
  const [readyAudioBlobUrl, setReadyAudioBlobUrl] = useState<string | null>(
    null
  );
  const [readyAudioEl, setReadyAudioEl] = useState<HTMLAudioElement | null>(
    null
  );
  const [isPlayingVoice, setIsPlayingVoice] = useState(false);
  const [audioDuration, setAudioDuration] = useState<number>(0);
  const [audioCurrentTime, setAudioCurrentTime] = useState<number>(0);

  const [activeMedicine, setActiveMedicine] =
    useState<MedicineDatabaseRecord | null>(null);
  const [loadingBrandId, setLoadingBrandId] = useState<string | null>(null);

  const webcamRef = useRef<Webcam>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const imagePreviewRef = useRef<string | null>(null);

  const hasUploadedImage = Boolean(selectedFile || imagePreview);
  const activeVoiceKey =
    step === "summary" && analysis?.audioUrl
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

  // Cleanup preview blob URL & audio on unmount
  useEffect(() => {
    imagePreviewRef.current = imagePreview;
  }, [imagePreview]);

  useEffect(() => {
    return () => {
      if (imagePreviewRef.current?.startsWith("blob:")) {
        URL.revokeObjectURL(imagePreviewRef.current);
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

  // Background voice pre-fetcher: as soon as scan results appear on screen, prepare the Bangla MP3
  // asynchronously while showing a live animated voice loader until ready.
  useEffect(() => {
    if (!activeVoiceKey || !analysis?.audioUrl) {
      return;
    }

    let cancelled = false;
    let createdObjUrl: string | null = null;
    const controller = new AbortController();

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
        const objUrl = URL.createObjectURL(blob);
        createdObjUrl = objUrl;
        setReadyAudioBlobUrl(objUrl);

        const audio = new Audio(objUrl);
        audioRef.current = audio;
        setReadyAudioEl(audio);
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
            setIsPlayingVoice(false);
            setAudioCurrentTime(audio.duration || 0);
          }
        };
        audio.onerror = () => {
          if (!cancelled) setIsPlayingVoice(false);
        };
        setVoiceLoadedForKey(activeVoiceKey);
      })
      .catch(() => {
        if (!cancelled) {
          setVoiceFailedForKey(activeVoiceKey);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setReadyAudioEl(null);
      setReadyAudioBlobUrl(null);
      if (createdObjUrl) {
        URL.revokeObjectURL(createdObjUrl);
      }
    };
  }, [activeVoiceKey, analysis?.audioUrl, analysis?.banglaVoiceScript]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImagePreview((prev) => {
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setSelectedFile(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    onDropRejected: () => {
      toast.error("File must be a JPEG, PNG, or WebP image under 15 MB.");
    },
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".webp"],
    },
    maxFiles: 1,
    maxSize: MAX_UPLOAD_BYTES,
    noClick: Boolean(imagePreview),
    noKeyboard: Boolean(imagePreview),
  });

  const handleWebcamSnap = async () => {
    const shot = webcamRef.current?.getScreenshot();
    if (shot) {
      setImagePreview((prev) => {
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
        return shot;
      });
      const res = await fetch(shot);
      const blob = await res.blob();
      setSelectedFile(
        new File([blob], "camera_capture.jpg", { type: "image/jpeg" })
      );
      setCaptureMode("file");
    }
  };

  const handleNextScan = useCallback(async () => {
    if (!selectedFile && !imagePreview) {
      toast.error("Please upload or capture a medicine photo first.");
      return;
    }

    setIsScanning(true);
    setScanStageIdx(0);
    setScanStageStatuses(["running", "pending", "pending", "pending", "pending"]);
    setScanStageDurations([null, null, null, null, null]);
    setScanElapsedMs(0);

    const t0 = Date.now();
    const timer = setInterval(() => {
      setScanElapsedMs(Math.round(Date.now() - t0));
    }, 100);

    try {
      let fileToAnalyze: File | Blob | null = selectedFile;
      if (!fileToAnalyze && imagePreview) {
        const res = await fetch(imagePreview);
        fileToAnalyze = await res.blob();
      }
      if (!fileToAnalyze) {
        toast.error("Please upload or capture a medicine photo first.");
        return;
      }

      const optimizedFile = await resizeImageForUpload(fileToAnalyze, 1920);
      const { data, isLiveBackend } = await analyzeMedicineImageStream(
        optimizedFile,
        (ev) => {
          if (ev.stageIndex < 0 || ev.stageIndex >= LIVE_SCAN_STAGES.length) {
            return;
          }
          setScanStageIdx(ev.stageIndex);
          setScanStageStatuses((prev) => {
            const next = [...prev];
            for (let i = 0; i < ev.stageIndex; i++) {
              next[i] = "done";
            }
            next[ev.stageIndex] = ev.status;
            return next;
          });
          if (ev.status === "done" && typeof ev.durationMs === "number") {
            setScanStageDurations((prev) => {
              const next = [...prev];
              next[ev.stageIndex] = ev.durationMs ?? null;
              return next;
            });
          }
        }
      );
      setScanStageStatuses(["done", "done", "done", "done", "done"]);
      setAnalysis(data);
      setReadyAudioBlobUrl(null);
      setAudioDuration(0);
      setAudioCurrentTime(0);
      setIsPlayingVoice(false);
      toast.success(
        isLiveBackend ? "FastAPI Scan Complete" : "Medicine Scan Complete",
        {
          description: `${data.detectedStrips.length} strips analyzed in ${data.inferenceTimeMs}ms`,
        }
      );
      setStep("summary");
    } catch (err) {
      toast.error("Medicine Scan Failed", {
        description:
          err instanceof Error
            ? err.message
            : "Could not analyze the uploaded image. Please try again.",
      });
    } finally {
      clearInterval(timer);
      setIsScanning(false);
    }
  }, [selectedFile, imagePreview]);

  const handleOpenStripDatabaseInfo = async (strip: BoundingBox) => {
    if (strip.databaseRecord) {
      setActiveMedicine({
        ...strip.databaseRecord,
        package_container:
          strip.strength ||
          strip.databaseRecord.package_container ||
          "Standard Dose",
      });
      return;
    }

    setLoadingBrandId(strip.id);
    try {
      const params = new URLSearchParams({
        q: strip.brandName,
        dosageForm: "All",
        page: "1",
        limit: "10",
      });
      const res = await fetch(`/api/medicines?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        const items: MedicineDatabaseRecord[] = data.items || [];
        const exact =
          items.find(
            (m) =>
              m.brand_name.toLowerCase() === strip.brandName.toLowerCase()
          ) || items[0];

        if (exact) {
          setActiveMedicine({
            ...exact,
            package_container:
              strip.strength || exact.package_container || "Standard Dose",
          });
          return;
        }
      }
    } catch (err) {
      console.error("Failed to fetch database record for strip:", err);
    } finally {
      setLoadingBrandId(null);
    }

    setActiveMedicine({
      brand_id: 0,
      brand_name: strip.brandName,
      type: "allopathic",
      dosage_form: strip.dosageForm || "Tablet",
      generic: strip.genericName,
      manufacturer: strip.manufacturer,
      package_container: strip.strength,
      drug_class: null,
      indication: strip.dosageEnglish,
      dosage_description: `${strip.dosageBangla}\n${strip.dosageEnglish}`,
      interaction_description: null,
      side_effects_description: null,
      pregnancy_and_lactation_description: null,
      beginner_summary: strip.dosageEnglish,
    });
  };

  const handleBackToUpload = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsPlayingVoice(false);
    setAudioCurrentTime(0);
    setStep("upload");
  };

  const handleResetFile = () => {
    setImagePreview((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setSelectedFile(null);
  };

  const toggleBanglaVoice = () => {
    if (typeof window === "undefined") return;
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

    if (!("speechSynthesis" in window)) return;
    setIsPlayingVoice(true);
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(analysis.banglaVoiceScript);
    utterance.lang = "bn-BD";
    utterance.rate = 0.95;
    utterance.onend = () => setIsPlayingVoice(false);
    utterance.onerror = () => setIsPlayingVoice(false);
    window.speechSynthesis.speak(utterance);
  };

  const audioProgressPct =
    audioDuration > 0
      ? Math.min(100, Math.round((audioCurrentTime / audioDuration) * 100))
      : 0;

  const activeStage = LIVE_SCAN_STAGES[scanStageIdx] || LIVE_SCAN_STAGES[0];
  const completedStagesCount = scanStageStatuses.filter(
    (st) => st === "done"
  ).length;
  const hasRunningStage = scanStageStatuses.some((st) => st === "running");
  const liveScanPct = Math.min(
    100,
    completedStagesCount * 20 + (hasRunningStage ? 10 : 0)
  );

  const steps = [
    { id: 1, label: "YOLO26", status: "done" },
    { id: 2, label: "PP-OCRv6", status: "done" },
    { id: 3, label: "17K+ BD Index", status: "done" },
    {
      id: 4,
      label: "Upload",
      status: step === "upload" ? "active" : "done",
    },
    {
      id: 5,
      label: "Summary",
      status: step === "summary" ? "active" : "upcoming",
    },
  ];

  const equalizerHeights = [40, 85, 55, 100, 65, 90, 45, 75, 60, 95, 50, 80];

  return (
    <section className="max-w-5xl mx-auto px-4 sm:px-6 pt-8 sm:pt-10 pb-20 space-y-8">
      {/* Horizontal 5-Step Progress Pill Bar (No duplicate logo here) */}
      <div className="rounded-full bg-white/95 backdrop-blur-xl p-2 shadow-[0_10px_35px_rgba(11,85,99,0.07)] border border-white overflow-x-auto">
        <div className="flex items-center justify-between gap-1.5 min-w-[460px]">
          {steps.map((s) => {
            const isDone = s.status === "done";
            const isActive = s.status === "active";

            const isStepDisabled =
              s.id === 5 && step === "upload" && (!hasUploadedImage || isScanning);

            return (
              <button
                key={s.id}
                type="button"
                disabled={isStepDisabled}
                onClick={() => {
                  if (s.id === 4) handleBackToUpload();
                  if (s.id === 5 && step === "upload" && hasUploadedImage) {
                    handleNextScan();
                  }
                }}
                className={`flex-1 flex items-center justify-between rounded-full px-3.5 py-2 text-xs font-semibold transition ${
                  isStepDisabled
                    ? "bg-[#f2f9f8]/60 text-[#46737a]/50 cursor-not-allowed"
                    : isActive
                    ? "bg-white text-[#09353c] shadow-sm border border-teal-200/70 cursor-pointer"
                    : "bg-[#f2f9f8] text-[#46737a] hover:bg-teal-50/70 cursor-pointer"
                }`}
              >
                <div className="flex items-center gap-2">
                  {isDone ? (
                    <span className="flex items-center justify-center size-5 rounded-full bg-[#0ea86b] text-white shrink-0">
                      <Check className="size-3 stroke-[3]" />
                    </span>
                  ) : isActive ? (
                    <span className="flex items-center justify-center size-5 rounded-full bg-[#0b5563] text-white text-[10px] font-bold shrink-0">
                      {s.id}
                    </span>
                  ) : (
                    <span className="flex items-center justify-center size-5 rounded-full bg-teal-100 text-[#46737a] text-[10px] font-bold shrink-0">
                      {s.id}
                    </span>
                  )}
                  <span
                    className={
                      isActive ? "font-bold text-[#09353c]" : "text-[#46737a]"
                    }
                  >
                    {s.label}
                  </span>
                </div>

                {isActive && (
                  <ChevronRight className="size-3.5 text-[#0d9488] shrink-0" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* LIVE 5-STAGE PROGRESS OVERLAY BANNER WHILE SCANNING */}
      {isScanning && (
        <div className="rounded-3xl bg-white/95 backdrop-blur-2xl p-6 shadow-xl border-2 border-[#0ea86b]/40 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <Loader2 className="size-5 text-[#0ea86b] animate-spin shrink-0" />
              <div>
                <p className="text-sm font-extrabold text-[#09353c]">
                  {activeStage.labelBn}
                </p>
                <p className="text-xs text-[#46737a]">{activeStage.labelEn}</p>
              </div>
            </div>
            <span className="font-mono text-xs font-extrabold text-[#0ea86b] bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 shrink-0">
              {liveScanPct}% • {scanElapsedMs} ms
            </span>
          </div>

          <div className="h-2.5 w-full rounded-full bg-teal-100 overflow-hidden">
            <div
              style={{ width: `${liveScanPct}%` }}
              className="h-full bg-gradient-to-r from-[#0b5563] to-[#0ea86b] transition-all duration-300 rounded-full"
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-1">
            {LIVE_SCAN_STAGES.map((st, idx) => {
              const status = scanStageStatuses[idx];
              const done = status === "done";
              const current = status === "running";
              const dur = scanStageDurations[idx];
              return (
                <div
                  key={st.id}
                  className={`rounded-xl px-2.5 py-2 text-[11px] font-bold flex items-center justify-between gap-1.5 border ${
                    current
                      ? "bg-emerald-50 text-[#09353c] border-[#0ea86b]"
                      : done
                      ? "bg-[#f2f9f8] text-[#0ea86b] border-teal-100"
                      : "bg-white text-[#46737a]/60 border-teal-50"
                  }`}
                >
                  <div className="flex items-center gap-1.5 min-w-0">
                    {done ? (
                      <Check className="size-3.5 text-[#0ea86b] shrink-0" />
                    ) : current ? (
                      <Loader2 className="size-3.5 text-[#0ea86b] animate-spin shrink-0" />
                    ) : (
                      <span className="size-3.5 rounded-full bg-teal-100 inline-block shrink-0" />
                    )}
                    <span className="truncate">ধাপ {st.id}/৫</span>
                  </div>
                  {dur !== null && (
                    <span className="font-mono text-[10px] shrink-0">
                      {dur}ms
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Main Centered White Card (Expanded to max-w-5xl like /database & /pipeline) */}
      <div className="rounded-[2.25rem] bg-white/95 backdrop-blur-2xl p-6 sm:p-10 shadow-[0_24px_70px_rgba(11,85,99,0.09)] border border-white">
        {step === "upload" ? (
          <div className="space-y-7 max-w-3xl mx-auto">
            {/* Title & Subtitle */}
            <div className="text-center space-y-1.5">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#09353c]">
                Upload Your Medicine Photo
              </h1>
              <p className="text-xs sm:text-sm text-[#46737a]">
                Place 1 to 6 medicine blister strips on a table for instant
                brand, strength (mg), Bangla summary, duplicate &amp;
                interaction screening.
              </p>
            </div>

            {/* Dashed Dropzone Box */}
            {captureMode === "file" ? (
              <div
                {...getRootProps()}
                className={`rounded-3xl border-2 border-dashed p-8 sm:p-12 text-center transition-all ${
                  !imagePreview ? "cursor-pointer" : ""
                } ${
                  isDragActive
                    ? "border-[#0ea86b] bg-emerald-50/40"
                    : "border-teal-200/90 bg-[#f2f9f8]/50 hover:border-[#0d9488]"
                }`}
              >
                <input {...getInputProps()} />

                {imagePreview ? (
                  <div className="space-y-4">
                    <div className="relative mx-auto max-w-md aspect-video rounded-2xl overflow-hidden border border-teal-200 shadow-sm bg-[#f2f9f8]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={imagePreview}
                        alt="Medicine strip preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex items-center justify-center gap-3">
                      <span className="text-xs font-bold text-[#09353c]">
                        {selectedFile?.name || "Captured Medicine Strip Photo"}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleResetFile();
                        }}
                        className="inline-flex items-center gap-1 text-xs font-bold text-[#0d9488] hover:underline cursor-pointer"
                      >
                        <RotateCcw className="size-3" />
                        Replace
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center space-y-3">
                    <div className="flex items-center justify-center size-12 rounded-2xl text-[#0b5563]">
                      <CloudUpload className="size-8 stroke-[1.75]" />
                    </div>

                    <div className="space-y-1">
                      <p className="text-base sm:text-lg font-bold text-[#09353c]">
                        Choose a file or drag &amp; drop it here.
                      </p>
                      <p className="text-xs sm:text-sm text-[#46737a]">
                        JPEG, PNG, and WebP formats, up to 15 MB.
                      </p>
                    </div>

                    <div className="pt-3 flex flex-wrap items-center justify-center gap-2.5">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          open();
                        }}
                        className="rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold px-7 py-2.5 border border-teal-200 shadow-2xs transition cursor-pointer active:scale-95"
                      >
                        Browse File
                      </button>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setCaptureMode("camera");
                        }}
                        className="inline-flex items-center gap-1.5 rounded-full bg-teal-50 hover:bg-teal-100/80 text-[#0b5563] text-xs sm:text-sm font-semibold px-5 py-2.5 transition cursor-pointer"
                      >
                        <Camera className="size-4 text-[#0ea86b]" />
                        <span>Use Camera</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Camera Capture Mode inside the Dashed Box */
              <div className="rounded-3xl border-2 border-dashed border-teal-200 p-6 text-center space-y-4 bg-[#f2f9f8]/50">
                <div className="relative mx-auto max-w-md aspect-video rounded-2xl overflow-hidden bg-[#09353c] border border-teal-200">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: "environment" }}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => setCaptureMode("file")}
                    className="rounded-full bg-white text-[#46737a] text-xs font-bold px-5 py-2.5 border border-teal-200 cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleWebcamSnap}
                    className="inline-flex items-center gap-1.5 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs font-bold px-6 py-2.5 shadow-sm cursor-pointer"
                  >
                    <Camera className="size-3.5" />
                    <span>Snap Photo</span>
                  </button>
                </div>
              </div>
            )}

            {/* Bottom Reset & Emerald Scan Pill Buttons */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <button
                type="button"
                onClick={handleResetFile}
                className="w-full rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold py-3.5 border border-teal-200 transition cursor-pointer"
              >
                Reset
              </button>

              <button
                type="button"
                disabled={isScanning || !hasUploadedImage}
                onClick={handleNextScan}
                title={
                  !hasUploadedImage
                    ? "Upload or capture a medicine strip photo first"
                    : "Scan uploaded photo"
                }
                className="w-full inline-flex items-center justify-center gap-2 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs sm:text-sm font-bold py-3.5 shadow-md shadow-[#0ea86b]/25 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-[#0ea86b]"
              >
                {isScanning ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    <span>Scanning ({liveScanPct}%)...</span>
                  </>
                ) : (
                  <span>
                    {hasUploadedImage
                      ? "Scan Uploaded Photo"
                      : "Upload Photo to Continue"}
                  </span>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* =============================================================
             STEP 5: SUMMARY VIEW (Structured Guide + Animated Audio Box)
             ============================================================= */
          <div className="space-y-6">
            <div className="text-center space-y-1">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 text-[#0ea86b] px-3 py-0.5 text-[11px] font-bold border border-emerald-200">
                <Check className="size-3 stroke-[3]" />
                Scan Completed in {analysis.inferenceTimeMs} ms
              </span>
              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#09353c]">
                Medicine Safety Summary (ওষুধের নিরাপত্তা রিপোর্ট)
              </h2>
              <p className="text-xs sm:text-sm text-[#46737a]">
                {analysis.detectedStrips.length}টি ওষুধের পাতা{" "}
                <code className="font-mono">medicines.db</code> থেকে যাচাইকৃত •
                যেকোনো ওষুধের কার্ডে ক্লিক করে সম্পূর্ণ ডাটাবেস তথ্য দেখুন
              </p>
            </div>

            {/* 1. DEDICATED ANIMATED BANGLA AUDIO PLAYER BOX (With Length & Progress Done) */}
            <div className="rounded-3xl bg-gradient-to-r from-[#09353c] via-[#0b5563] to-[#0d6e6e] p-5 sm:p-6 text-white shadow-lg border border-teal-700/40 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3.5">
                  <button
                    type="button"
                    onClick={toggleBanglaVoice}
                    disabled={isVoiceLoading}
                    className={`flex items-center justify-center size-12 rounded-2xl shadow-md transition cursor-pointer shrink-0 ${
                      isVoiceLoading
                        ? "bg-white/15 text-emerald-300 cursor-wait"
                        : isPlayingVoice
                        ? "bg-white text-[#09353c] hover:bg-emerald-50"
                        : "bg-[#0ea86b] text-white hover:bg-emerald-400"
                    }`}
                    aria-label={isPlayingVoice ? "Pause Bangla Audio" : "Play Bangla Audio"}
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
                        বাংলা ভয়েস গাইড অডিও (Bangla Voice Player)
                      </span>
                      {isVoiceLoading ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-400/20 text-amber-200 border border-amber-300/30 px-2.5 py-0.5 text-[10px] font-bold">
                          <Loader2 className="size-3 animate-spin" />
                          অডিও তৈরি হচ্ছে...
                        </span>
                      ) : isVoiceReady ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-400/20 text-emerald-200 border border-emerald-300/30 px-2.5 py-0.5 text-[10px] font-bold">
                          <Check className="size-3 stroke-[3]" />
                          {isPlayingVoice ? "চলছে (Playing)" : "প্রস্তুত (Ready)"}
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-white/15 text-teal-100 border border-white/20 px-2.5 py-0.5 text-[10px] font-bold">
                          ব্রাউজার ভয়েস (Fallback)
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-teal-100/85">
                      {isVoiceLoading
                        ? "অন্যান্য তথ্য নিচে দেখানো হয়েছে: ভয়েস অডিও ব্যাকগ্রাউন্ডে প্রস্তুত হচ্ছে..."
                        : `মোট দৈর্ঘ্য: ${formatAudioTime(audioDuration)} • সম্পন্ন: ${audioProgressPct}%`}
                    </p>
                  </div>
                </div>

                {/* Animated Soundwave Equalizer Bars */}
                <div className="flex items-end justify-center sm:justify-end gap-1 h-9 px-3 py-1.5 rounded-2xl bg-white/10 border border-white/10">
                  {equalizerHeights.map((h, i) => (
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

              {/* WaveSurfer.js Interactive Bangla Voice Waveform & Progress */}
              <div className="space-y-1.5">
                <BanglaVoiceWaveform
                  audioUrl={readyAudioBlobUrl}
                  audioElement={readyAudioEl}
                  isLoading={isVoiceLoading}
                  onSeek={setAudioCurrentTime}
                />
                <div className="flex items-center justify-between text-[11px] font-mono text-teal-100/90">
                  <span>{formatAudioTime(audioCurrentTime)}</span>
                  <span className="font-bold text-emerald-300">
                    {isVoiceLoading
                      ? "Generating MP3..."
                      : `${audioProgressPct}% Done`}
                  </span>
                  <span>
                    {audioDuration > 0
                      ? formatAudioTime(audioDuration)
                      : "--:--"}
                  </span>
                </div>
              </div>
            </div>

            {/* 1B. BANGLA VOICE GUIDE & LINE-BY-LINE SUMMARY (Matches /pipeline) */}
            {analysis.banglaVoiceScript && (
              <div className="rounded-3xl bg-[#f2f9f8] p-5 border border-teal-100 space-y-4">
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

                <ul className="space-y-2">
                  {splitBanglaSummaryLines(analysis.banglaVoiceScript).map(
                    (line, i) => {
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
                              : "font-medium text-[#09353c] pl-1"
                          }`}
                        >
                          {isStripLine && (
                            <span className="mt-1.5 size-2 rounded-full bg-[#0ea86b] shrink-0" />
                          )}
                          <span>{line}</span>
                        </li>
                      );
                    }
                  )}
                </ul>
              </div>
            )}

            {/* 2. CLEANLY ALIGNED STRUCTURED BANGLA GUIDE SUMMARY BOX */}
            <div className="rounded-3xl bg-[#f2f9f8] p-5 border border-teal-100 space-y-3.5">
              <div className="flex items-center justify-between border-b border-teal-200/60 pb-2.5">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563] flex items-center gap-1.5">
                  <BookOpen className="size-4 text-[#0ea86b]" />
                  বাংলা গাইড সারসংক্ষেপ (Quick Medicine Overview)
                </p>
                <span className="rounded-full bg-white px-2.5 py-0.5 text-[10px] font-bold text-[#0ea86b] border border-teal-100">
                  {analysis.detectedStrips.length}টি শনাক্তকৃত ওষুধ
                </span>
              </div>

              <div className="divide-y divide-teal-100/80">
                {analysis.detectedStrips.map((strip, idx) => (
                  <div
                    key={strip.id}
                    className="py-2.5 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                  >
                    <div className="flex items-start gap-2.5">
                      <span className="flex items-center justify-center size-6 rounded-full bg-[#0ea86b] text-white text-[11px] font-extrabold shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <div className="space-y-0.5">
                        <p className="text-xs sm:text-sm font-extrabold text-[#09353c]">
                          {strip.brandName}{" "}
                          <span className="text-[#0ea86b]">
                            ({strip.strength})
                          </span>
                          <span className="ml-2 font-semibold text-xs text-[#0d9488]">
                            • {strip.genericName}
                          </span>
                        </p>
                        <p className="text-xs text-[#46737a] leading-relaxed">
                          {strip.dosageBangla}
                        </p>
                      </div>
                    </div>
                    <span
                      className={`self-start sm:self-center rounded-full px-2.5 py-0.5 text-[10px] font-extrabold shrink-0 border ${
                        strip.status === "duplicate"
                          ? "bg-rose-50 text-rose-700 border-rose-200"
                          : strip.status === "interaction"
                          ? "bg-amber-50 text-amber-800 border-amber-200"
                          : "bg-emerald-50 text-[#0ea86b] border-emerald-200"
                      }`}
                    >
                      {strip.status === "duplicate"
                        ? "ডুপ্লিকেট সতর্কতা"
                        : strip.status === "interaction"
                        ? "সতর্কতা"
                        : "নিরাপদ"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* 3. Duplicate & Interaction Alerts */}
            {analysis.alerts.length > 0 && (
              <div className="space-y-2.5">
                {analysis.alerts.map((alert) => {
                  const isDup = alert.type === "duplicate";
                  const isSafe = alert.type === "safe";
                  return (
                    <div
                      key={alert.id}
                      className={`rounded-2xl p-4 border ${
                        isDup
                          ? "bg-rose-50/70 border-rose-200"
                          : isSafe
                          ? "bg-emerald-50/70 border-emerald-200"
                          : "bg-teal-50/70 border-teal-200"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {isDup ? (
                          <AlertOctagon className="size-4 text-rose-600 shrink-0" />
                        ) : isSafe ? (
                          <CheckCircle2 className="size-4 text-[#0ea86b] shrink-0" />
                        ) : (
                          <AlertTriangle className="size-4 text-[#0b5563] shrink-0" />
                        )}
                        <h3 className="font-bold text-xs text-[#09353c]">
                          {alert.titleBangla}
                        </h3>
                      </div>
                      <p className="text-xs text-[#46737a] mt-1 leading-relaxed">
                        {alert.descriptionBangla}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}

            {/* 4. Detected Strips Database Cards (Click any card to open full Database Modal; no সময়সূচী row) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {analysis.detectedStrips.map((strip) => {
                const isDup = strip.status === "duplicate";
                const dbRec = strip.databaseRecord;
                const dosageForm =
                  strip.dosageForm || dbRec?.dosage_form || "Tablet";
                const indication = dbRec?.indication || null;
                const banglaSummary =
                  dbRec?.beginner_summary ||
                  strip.dosageEnglish ||
                  strip.dosageBangla;
                const isLoadingThis = loadingBrandId === strip.id;

                return (
                  <div
                    key={strip.id}
                    onClick={() => handleOpenStripDatabaseInfo(strip)}
                    className={`group rounded-2xl bg-white p-5 border transition-all flex flex-col justify-between cursor-pointer hover:-translate-y-0.5 ${
                      isDup
                        ? "border-rose-200 shadow-[0_8px_25px_rgba(225,29,72,0.08)] hover:shadow-[0_14px_35px_rgba(225,29,72,0.16)]"
                        : "border-teal-100 shadow-[0_8px_25px_rgba(11,85,99,0.06)] hover:shadow-[0_14px_35px_rgba(14,168,107,0.16)]"
                    }`}
                  >
                    <div className="space-y-3">
                      {/* Top Row: Form Badge + Extracted Strength Pill */}
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <div className="flex items-center justify-center size-8 rounded-full bg-[#f2f9f8] border border-teal-100 shrink-0">
                            <Pill className="size-4 text-[#0ea86b]" />
                          </div>
                          <span className="font-mono text-[10px] font-bold text-[#0b5563] bg-[#f2f9f8] border border-teal-100 px-2.5 py-0.5 rounded-full">
                            {dosageForm}
                          </span>
                        </div>

                        <span
                          className={`rounded-full px-2.5 py-0.5 text-[11px] font-extrabold ${
                            isDup
                              ? "bg-rose-100 text-rose-700 border border-rose-200"
                              : "bg-emerald-50 text-[#0ea86b] border border-emerald-200"
                          }`}
                        >
                          {strip.strength}
                        </span>
                      </div>

                      {/* Brand Name & Generic Name */}
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <h3 className="font-extrabold text-base text-[#09353c] group-hover:text-[#0ea86b] transition">
                            {strip.brandName}{" "}
                            <span className="text-sm font-bold text-[#0b5563]">
                              ({strip.strength})
                            </span>
                          </h3>
                          {isLoadingThis ? (
                            <Loader2 className="size-4 text-[#0ea86b] animate-spin shrink-0" />
                          ) : (
                            <ArrowUpRight className="size-4 text-[#0d9488] opacity-70 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition shrink-0" />
                          )}
                        </div>
                        <p className="text-xs font-extrabold text-[#0d9488] mt-0.5 line-clamp-1">
                          {strip.genericName}
                        </p>
                      </div>

                      {/* Metadata Rows + Bangla Beginner Summary */}
                      <div className="pt-2.5 border-t border-teal-50 space-y-1.5 text-[11px]">
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-[#46737a] shrink-0">
                            প্রস্তুতকারক:
                          </span>
                          <span className="font-semibold text-[#09353c] text-right line-clamp-1">
                            {strip.manufacturer || "Pharmaceuticals BD"}
                          </span>
                        </div>

                        {indication && (
                          <div className="pt-1">
                            <span className="inline-block rounded-lg bg-[#f2f9f8] px-2.5 py-0.5 text-[10px] font-semibold text-[#0d9488] line-clamp-1">
                              Indication: {indication}
                            </span>
                          </div>
                        )}

                        {banglaSummary && (
                          <div className="rounded-xl bg-[#f2f9f8]/90 p-2.5 border border-teal-100/70 mt-1.5">
                            <span className="block text-[10px] font-extrabold text-[#0ea86b] mb-0.5">
                              সহজ বাংলা সারসংক্ষেপ (Gemma 3 1B):
                            </span>
                            <p className="text-[11px] text-[#09353c] leading-relaxed whitespace-pre-line">
                              {banglaSummary}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Footer Action Bar */}
                    <div className="mt-3 pt-2.5 border-t border-teal-100/70 flex items-center justify-between text-[11px] font-bold text-[#0ea86b]">
                      <span className="inline-flex items-center gap-1">
                        <Database className="size-3.5" />
                        বিস্তারিত ডাটাবেস তথ্য দেখুন
                      </span>
                      <span className="text-[10px] text-[#46737a] font-semibold">
                        Click →
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom Back, AI Pipeline Page Link & Scan Another Photo Buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
              <button
                type="button"
                onClick={handleBackToUpload}
                className="w-full rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold py-3.5 border border-teal-200 transition cursor-pointer"
              >
                Back
              </button>

              <Link
                href="/pipeline"
                className="w-full inline-flex items-center justify-center gap-1.5 rounded-full bg-[#f2f9f8] hover:bg-teal-100/80 text-[#0b5563] text-xs sm:text-sm font-bold py-3.5 border border-teal-200 transition"
              >
                <Cpu className="size-4 text-[#0ea86b]" />
                <span>AI Pipeline Inspector</span>
              </Link>

              <button
                type="button"
                onClick={() => {
                  handleResetFile();
                  handleBackToUpload();
                }}
                className="w-full rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs sm:text-sm font-bold py-3.5 shadow-md shadow-[#0ea86b]/20 transition cursor-pointer"
              >
                Scan New Photo
              </button>
            </div>
          </div>
        )}
      </div>

      {/* =====================================================================
          CLINICAL DATABASE DETAIL MODAL (With Bangla Beginner Summary)
         ===================================================================== */}
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
            {/* Modal Header */}
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

            {/* Top 2-Column Metadata Cards */}
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

            {/* Bangla Beginner-Friendly Summary Highlight Card */}
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

            {/* Primary Indication */}
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

            {/* Structured Dosage & Administration with Separated Sub-Cards */}
            {activeMedicine.dosage_description && (
              <div className="rounded-3xl bg-white p-5 border border-teal-200/80 shadow-2xs space-y-3.5">
                <div className="flex items-center justify-between border-b border-teal-100 pb-2.5">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                    <Clock className="size-4 text-[#0ea86b]" />
                    Dosage &amp; Administration
                  </p>
                  <span className="text-[10px] font-semibold text-[#46737a] bg-[#f2f9f8] px-2.5 py-0.5 rounded-full">
                    Clinical Regimen
                  </span>
                </div>

                <div className="space-y-3">
                  {parseStructuredClinicalText(
                    activeMedicine.dosage_description
                  ).map((item, idx) => (
                    <div
                      key={idx}
                      className="rounded-2xl bg-[#f2f9f8]/80 p-3.5 border border-teal-100/70 space-y-1.5"
                    >
                      {item.label && (
                        <span className="inline-block rounded-lg bg-[#0b5563] text-white px-2.5 py-0.5 text-[11px] font-bold">
                          {item.label}
                        </span>
                      )}
                      <p className="text-xs sm:text-sm text-[#09353c] leading-relaxed">
                        {item.content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Drug-Drug Interactions */}
            {activeMedicine.interaction_description && (
              <div className="rounded-3xl bg-amber-50/70 p-5 border border-amber-200/90 space-y-2.5">
                <p className="text-xs font-extrabold uppercase tracking-wider text-amber-800 flex items-center gap-1.5">
                  <AlertTriangle className="size-4 text-amber-600" />
                  Drug-Drug Interactions
                </p>
                <div className="space-y-2">
                  {parseStructuredClinicalText(
                    activeMedicine.interaction_description
                  ).map((item, idx) => (
                    <div
                      key={idx}
                      className="rounded-2xl bg-white/80 p-3 border border-amber-200/60 space-y-1"
                    >
                      {item.label && (
                        <span className="inline-block rounded-md bg-amber-100 text-amber-900 px-2 py-0.5 text-[11px] font-bold">
                          {item.label}
                        </span>
                      )}
                      <p className="text-xs sm:text-sm text-amber-950/90 leading-relaxed">
                        {item.content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Side Effects */}
            {activeMedicine.side_effects_description && (
              <div className="rounded-3xl bg-white p-5 border border-teal-100 space-y-2.5">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                  Side Effects &amp; Precautions
                </p>
                <p className="text-xs sm:text-sm text-[#46737a] leading-relaxed">
                  {activeMedicine.side_effects_description}
                </p>
              </div>
            )}

            {/* Pregnancy & Lactation */}
            {activeMedicine.pregnancy_and_lactation_description && (
              <div className="rounded-3xl bg-[#f2f9f8]/60 p-5 border border-teal-100 space-y-2">
                <p className="text-xs font-extrabold uppercase tracking-wider text-[#0b5563]">
                  Pregnancy &amp; Lactation
                </p>
                <p className="text-xs sm:text-sm text-[#46737a] leading-relaxed">
                  {activeMedicine.pregnancy_and_lactation_description}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
