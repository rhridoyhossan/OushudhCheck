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
} from "lucide-react";
import { PRESET_SCENARIOS } from "@/lib/mock-data";
import { ScanAnalysisResult } from "@/lib/types";
import { analyzeMedicineImage } from "@/lib/api";
import { toast } from "sonner";

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

  const webcamRef = useRef<Webcam>(null);

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

  const handleBackToUpload = () => {
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

  const audioRef = useRef<HTMLAudioElement | null>(null);

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
      const baseUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1").replace(/\/api\/v1\/?$/, "");
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
                Place 2 to 6 medicine blister strips on a table for instant duplicate
                &amp; interaction screening.
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
             STEP 5: SUMMARY VIEW (Logo Teal & Emerald Theme)
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
                {analysis.detectedStrips.length} strips resolved against 21,000+
                Bangladeshi brands
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

            {/* Detected Strips 2x2 Minimal Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {analysis.detectedStrips.map((strip) => {
                const isDup = strip.status === "duplicate";
                return (
                  <div
                    key={strip.id}
                    className="rounded-2xl bg-[#f2f9f8] p-3.5 border border-teal-100 flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-xs text-[#09353c] flex items-center gap-1.5">
                        <Pill className="size-3.5 text-[#0ea86b]" />
                        {strip.brandName} {strip.strength}
                      </span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                          isDup
                            ? "bg-rose-100 text-rose-700"
                            : "bg-emerald-100 text-[#0b5563]"
                        }`}
                      >
                        {strip.genericName}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#46737a] mt-1.5">
                      {strip.dosageBangla}
                    </p>
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
    </section>
  );
}
