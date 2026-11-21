"use client";

import React, { useRef, useState, useCallback } from "react";
import Image from "next/image";
import Webcam from "react-webcam";
import { useDropzone } from "react-dropzone";
import {
  CloudUpload,
  Check,
  ChevronRight,
  Camera,
  Volume2,
  Pause,
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
} from "lucide-react";
import { PRESET_SCENARIOS } from "@/lib/mock-data";
import {
  BoundingBox,
  MedicineDatabaseRecord,
  ScanAnalysisResult,
} from "@/lib/types";
import { analyzeMedicineImage } from "@/lib/api";
import { toast } from "sonner";

/**
 * Parses a continuous clinical text block (such as `dosage_description` from medicines.db)
 * into clean, well-spaced sections (`{ label, content }`) matching the Database Directory modal.
 */
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
  const [analysis, setAnalysis] = useState<ScanAnalysisResult>(
    PRESET_SCENARIOS[0].result
  );
  const [isPlayingVoice, setIsPlayingVoice] = useState(false);
  const [activeMedicine, setActiveMedicine] =
    useState<MedicineDatabaseRecord | null>(null);
  const [loadingBrandId, setLoadingBrandId] = useState<string | null>(null);

  const webcamRef = useRef<Webcam>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".webp"],
    },
    maxFiles: 1,
    noClick: true,
  });

  const handleWebcamSnap = async () => {
    const shot = webcamRef.current?.getScreenshot();
    if (shot) {
      setImagePreview(shot);
      const res = await fetch(shot);
      const blob = await res.blob();
      setSelectedFile(
        new File([blob], "camera_capture.jpg", { type: "image/jpeg" })
      );
      setCaptureMode("file");
    }
  };

  const handleNextScan = async () => {
    setIsScanning(true);

    let fileToAnalyze: File | Blob | null = selectedFile;
    if (!fileToAnalyze && imagePreview) {
      const res = await fetch(imagePreview);
      fileToAnalyze = await res.blob();
    }

    if (fileToAnalyze) {
      const { data, isLiveBackend } = await analyzeMedicineImage(fileToAnalyze);
      setAnalysis(data);
      toast.success(
        isLiveBackend ? "FastAPI Scan Complete" : "Medicine Scan Complete",
        {
          description: `${data.detectedStrips.length} strips analyzed in ${data.inferenceTimeMs}ms`,
        }
      );
    } else {
      await new Promise((r) => setTimeout(r, 650));
      setAnalysis(PRESET_SCENARIOS[0].result);
      toast.success("Medicine Scan Complete", {
        description: `${PRESET_SCENARIOS[0].result.detectedStrips.length} strips analyzed`,
      });
    }

    setIsScanning(false);
    setStep("summary");
  };

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

    // Fallback record constructed from strip fields if database query didn't return
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
      dosage_description: `${strip.dosageEnglish}\n${strip.dosageBangla}`,
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
    setStep("upload");
  };

  const handleResetFile = () => {
    setSelectedFile(null);
    setImagePreview(null);
  };

  const toggleBanglaVoice = () => {
    if (typeof window === "undefined") return;
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

    if (analysis.audioUrl) {
      const baseUrl = (
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1"
      ).replace(/\/api\/v1\/?$/, "");
      const fullAudioUrl = analysis.audioUrl.startsWith("http")
        ? analysis.audioUrl
        : `${baseUrl}${analysis.audioUrl}`;
      const audio = new Audio(fullAudioUrl);
      audioRef.current = audio;
      setIsPlayingVoice(true);
      audio.onended = () => setIsPlayingVoice(false);
      audio.onerror = () => {
        setIsPlayingVoice(false);
      };
      audio.play().catch(() => setIsPlayingVoice(false));
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

  const steps = [
    { id: 1, label: "YOLO26", status: "done" },
    { id: 2, label: "Foil OCR", status: "done" },
    { id: 3, label: "21K+ BD Index", status: "done" },
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

  return (
    <section className="max-w-3xl mx-auto px-4 sm:px-6 pt-6 pb-20">
      {/* Top Centered Brand Header */}
      <div className="flex items-center justify-center gap-2.5 mb-6">
        <div className="relative size-8 rounded-xl overflow-hidden bg-white shadow-sm border border-white shrink-0">
          <Image
            src="/logo.jpg"
            alt="OushudhCheck Logo"
            fill
            className="object-cover"
          />
        </div>
        <span className="text-xl font-extrabold tracking-tight text-[#09353c]">
          Oushudh<span className="text-[#0ea86b]">Check</span>
        </span>
      </div>

      {/* Horizontal 5-Step Progress Pill Bar */}
      <div className="rounded-full bg-white/95 backdrop-blur-xl p-2 shadow-[0_10px_35px_rgba(11,85,99,0.07)] border border-white mb-6 overflow-x-auto">
        <div className="flex items-center justify-between gap-1.5 min-w-[460px]">
          {steps.map((s) => {
            const isDone = s.status === "done";
            const isActive = s.status === "active";

            return (
              <button
                key={s.id}
                type="button"
                onClick={() => {
                  if (s.id === 4) handleBackToUpload();
                  if (s.id === 5 && step === "upload") handleNextScan();
                }}
                className={`flex-1 flex items-center justify-between rounded-full px-3.5 py-2 text-xs font-semibold transition cursor-pointer ${
                  isActive
                    ? "bg-white text-[#09353c] shadow-sm border border-teal-200/70"
                    : "bg-[#f2f9f8] text-[#46737a] hover:bg-teal-50/70"
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

      {/* Main Centered White Card */}
      <div className="rounded-[2.25rem] bg-white/95 backdrop-blur-2xl p-6 sm:p-10 shadow-[0_24px_70px_rgba(11,85,99,0.09)] border border-white">
        {step === "upload" ? (
          <div className="space-y-7">
            {/* Title & Subtitle */}
            <div className="text-center space-y-1.5">
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[#09353c]">
                Upload Your Medicine Photo
              </h1>
              <p className="text-xs sm:text-sm text-[#46737a]">
                Place 1 to 6 medicine blister strips on a table for instant
                brand, strength (mg), duplicate &amp; interaction screening.
              </p>
            </div>

            {/* Dashed Dropzone Box */}
            {captureMode === "file" ? (
              <div
                {...getRootProps()}
                className={`rounded-3xl border-2 border-dashed p-8 sm:p-12 text-center transition-all ${
                  isDragActive
                    ? "border-[#0ea86b] bg-emerald-50/40"
                    : "border-teal-200/90 bg-[#f2f9f8]/50 hover:border-[#0d9488]"
                }`}
              >
                <input {...getInputProps()} />

                {imagePreview ? (
                  <div className="space-y-4">
                    <div className="relative mx-auto max-w-sm aspect-video rounded-2xl overflow-hidden border border-teal-200 shadow-sm bg-[#f2f9f8]">
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
                        onClick={handleResetFile}
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
                        JPEG, PNG, and WebP formats, up to 50 MB.
                      </p>
                    </div>

                    <div className="pt-3 flex flex-wrap items-center justify-center gap-2.5">
                      <button
                        type="button"
                        onClick={open}
                        className="rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold px-7 py-2.5 border border-teal-200 shadow-2xs transition cursor-pointer active:scale-95"
                      >
                        Browse File
                      </button>

                      <button
                        type="button"
                        onClick={() => setCaptureMode("camera")}
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

            {/* Bottom Back & Emerald Next Pill Buttons */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <button
                type="button"
                onClick={handleResetFile}
                className="w-full rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold py-3.5 border border-teal-200 transition cursor-pointer"
              >
                Back
              </button>

              <button
                type="button"
                disabled={isScanning}
                onClick={handleNextScan}
                className="w-full inline-flex items-center justify-center gap-2 rounded-full bg-[#0ea86b] hover:bg-[#0b5563] text-white text-xs sm:text-sm font-bold py-3.5 shadow-md shadow-[#0ea86b]/25 transition cursor-pointer disabled:opacity-60"
              >
                {isScanning ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    <span>Scanning...</span>
                  </>
                ) : (
                  <span>Next</span>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* =============================================================
             STEP 5: SUMMARY VIEW WITH CLICKABLE DATABASE CARDS & MODAL
             ============================================================= */
          <div className="space-y-6">
            <div className="text-center space-y-1">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 text-[#0ea86b] px-3 py-0.5 text-[11px] font-bold border border-emerald-200">
                <Check className="size-3 stroke-[3]" />
                Scan Completed in {analysis.inferenceTimeMs} ms
              </span>
              <h2 className="text-2xl font-extrabold tracking-tight text-[#09353c]">
                Medicine Safety Summary
              </h2>
              <p className="text-xs text-[#46737a]">
                {analysis.detectedStrips.length} medicine strip(s) matched against{" "}
                <code className="font-mono">medicines.db</code> • Click any
                medicine card below for full clinical &amp; database details
              </p>
            </div>

            {/* Bangla Voice Readout Bar */}
            <div className="rounded-2xl bg-[#f2f9f8] p-4 border border-teal-100 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="space-y-1 text-center sm:text-left">
                <p className="text-[11px] font-bold uppercase tracking-wider text-[#0d9488]">
                  বাংলা ভয়েস গাইড (Bangla Audio Guidance)
                </p>
                <p className="text-xs sm:text-sm font-semibold text-[#09353c] leading-relaxed">
                  “{analysis.banglaVoiceScript}”
                </p>
              </div>

              <button
                type="button"
                onClick={toggleBanglaVoice}
                className={`shrink-0 inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-xs font-bold text-white shadow-sm transition cursor-pointer ${
                  isPlayingVoice
                    ? "bg-[#0b5563] hover:bg-[#09353c]"
                    : "bg-[#0ea86b] hover:bg-[#0b5563]"
                }`}
              >
                {isPlayingVoice ? (
                  <>
                    <Pause className="size-3.5 fill-current" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Volume2 className="size-4" />
                    <span>বাংলায় শুনুন</span>
                  </>
                )}
              </button>
            </div>

            {/* Duplicate & Interaction Alerts */}
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

            {/* Detected Strips Database Cards (Click any card to open full Database Modal) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {analysis.detectedStrips.map((strip) => {
                const isDup = strip.status === "duplicate";
                const dbRec = strip.databaseRecord;
                const dosageForm =
                  strip.dosageForm || dbRec?.dosage_form || "Tablet";
                const indication = dbRec?.indication || null;
                const beginnerSummary =
                  dbRec?.beginner_summary || strip.dosageEnglish;
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
                          <h3 className="font-extrabold text-base sm:text-lg text-[#09353c] group-hover:text-[#0ea86b] transition">
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

                      {/* Metadata Rows matching Database Section */}
                      <div className="pt-2.5 border-t border-teal-50 space-y-1.5 text-[11px]">
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-[#46737a] shrink-0">
                            Company:
                          </span>
                          <span className="font-semibold text-[#09353c] text-right line-clamp-1">
                            {strip.manufacturer || "Pharmaceuticals BD"}
                          </span>
                        </div>

                        {dbRec?.drug_class && (
                          <div className="flex items-start justify-between gap-2">
                            <span className="text-[#46737a] shrink-0">
                              Class:
                            </span>
                            <span className="font-medium text-[#0b5563] text-right line-clamp-1">
                              {dbRec.drug_class}
                            </span>
                          </div>
                        )}

                        {indication && (
                          <div className="pt-1">
                            <span className="inline-block rounded-lg bg-[#f2f9f8] px-2.5 py-0.5 text-[10px] font-semibold text-[#0d9488] line-clamp-1">
                              Indication: {indication}
                            </span>
                          </div>
                        )}

                        {beginnerSummary && (
                          <p className="text-[11px] text-[#46737a] line-clamp-2 pt-1 leading-relaxed">
                            {beginnerSummary}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Footer Action Bar */}
                    <div className="mt-3 pt-2.5 border-t border-teal-100/70 flex items-center justify-between text-[11px] font-bold text-[#0ea86b]">
                      <span className="inline-flex items-center gap-1">
                        <Database className="size-3.5" />
                        View Full Database Info
                      </span>
                      <span className="text-[10px] text-[#46737a] font-semibold">
                        Click Card →
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Bottom Back & Scan Another Photo Buttons */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <button
                type="button"
                onClick={handleBackToUpload}
                className="w-full rounded-full bg-white hover:bg-[#f2f9f8] text-[#09353c] text-xs sm:text-sm font-bold py-3.5 border border-teal-200 transition cursor-pointer"
              >
                Back
              </button>

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
          CLINICAL DATABASE DETAIL MODAL (Identical to Database Section Modal)
         ===================================================================== */}
      {activeMedicine && (
        <div
          onClick={() => setActiveMedicine(null)}
          className="fixed inset-0 z-50 bg-[#09353c]/55 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl max-h-[86vh] overflow-y-auto rounded-[2.25rem] bg-white p-6 sm:p-8 shadow-2xl border border-teal-100 space-y-6"
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

            {/* Beginner-Friendly Summary Highlight Card */}
            {activeMedicine.beginner_summary && (
              <div className="rounded-3xl bg-gradient-to-br from-[#f2f9f8] via-emerald-50/60 to-[#f2f9f8] p-5 border-2 border-[#0ea86b]/30 shadow-xs space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-extrabold uppercase tracking-wider text-[#0ea86b] flex items-center gap-1.5">
                    <Pill className="size-4 text-[#0ea86b]" />
                    Beginner-Friendly Summary (Plain English)
                  </p>
                  <span className="rounded-full bg-white px-2.5 py-0.5 text-[10px] font-bold text-[#0b5563] border border-teal-100">
                    Easy Guide
                  </span>
                </div>
                <p className="text-xs sm:text-sm font-medium text-[#09353c] leading-relaxed">
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
