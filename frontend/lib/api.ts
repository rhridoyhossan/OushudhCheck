import { BenchmarkSample, ScanAnalysisResult } from "./types";
import { PRESET_SCENARIOS } from "./mock-data";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export const BACKEND_ORIGIN = API_BASE_URL.replace(/\/api(?:\/v1)?\/?$/, "");

/**
 * Fetches synthesized Bangla MP3 audio from FastAPI.
 * Tries the short cached clip URL first, then falls back cleanly to `POST /api/v1/tts`
 * with a JSON body so full multi-strip Bangla summaries (up to 6,000 chars) are never
 * truncated by URL length limits.
 */
export async function fetchBanglaTtsAudioBlob(
  text: string,
  audioUrl?: string | null,
  signal?: AbortSignal
): Promise<Blob> {
  if (audioUrl) {
    const fullAudioUrl = audioUrl.startsWith("http")
      ? audioUrl
      : `${BACKEND_ORIGIN}${audioUrl}`;
    try {
      const clipRes = await fetch(fullAudioUrl, { signal });
      if (clipRes.ok) {
        return await clipRes.blob();
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw err;
      }
    }
  }

  const trimmed = (text || "").trim();
  if (!trimmed) {
    throw new Error("No Bangla text provided for TTS synthesis.");
  }

  const postRes = await fetch(`${API_BASE_URL}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: trimmed }),
    signal,
  });

  if (!postRes.ok) {
    throw new Error(`Bangla TTS failed with HTTP ${postRes.status}`);
  }

  return await postRes.blob();
}

/**
 * Sends a captured or uploaded medicine strip image to the FastAPI backend
 * (`/api/v1/scan`). Throws a descriptive error if the scan fails so real user
 * photos are never silently replaced with unrelated mock medicines.
 */
export async function analyzeMedicineImage(
  fileOrBlob: File | Blob
): Promise<{ data: ScanAnalysisResult; isLiveBackend: boolean }> {
  const formData = new FormData();
  const fileName =
    fileOrBlob instanceof File && fileOrBlob.name
      ? fileOrBlob.name
      : "tabletop_strips.jpg";
  formData.append("file", fileOrBlob, fileName);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45000);

  try {
    const response = await fetch(`${API_BASE_URL}/scan`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Scan failed with HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        if (errBody?.detail) detail = String(errBody.detail);
      } catch {
        // ignore JSON parse failure
      }
      throw new Error(detail);
    }

    const json = await response.json();
    return { data: json as ScanAnalysisResult, isLiveBackend: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Scan timed out. Please ensure the FastAPI backend is running and try again.");
    }
    throw err instanceof Error
      ? err
      : new Error("Unable to reach the OushudhCheck backend server.");
  } finally {
    clearTimeout(timeout);
  }
}

export interface ScanStageProgressEvent {
  stageIndex: number;
  status: "running" | "done";
  durationMs?: number;
}

/**
 * Streams real-time per-stage progress events from FastAPI (`POST /api/v1/scan-stream`)
 * as each AI stage actually starts and completes, then resolves with the final ScanAnalysisResult.
 * Falls back to `POST /api/v1/scan` if streaming is unavailable.
 */
export interface ScanDetectorOptions {
  detectorMode?: "hybrid_fusion" | "yolo_only" | "ocr_only";
  yoloConf?: number;
}

export async function analyzeMedicineImageStream(
  fileOrBlob: File | Blob,
  onProgress?: (event: ScanStageProgressEvent) => void,
  options?: ScanDetectorOptions
): Promise<{ data: ScanAnalysisResult; isLiveBackend: boolean }> {
  const formData = new FormData();
  const fileName =
    fileOrBlob instanceof File && fileOrBlob.name
      ? fileOrBlob.name
      : "tabletop_strips.jpg";
  formData.append("file", fileOrBlob, fileName);

  const params = new URLSearchParams();
  if (options?.detectorMode) params.set("detector_mode", options.detectorMode);
  if (typeof options?.yoloConf === "number") {
    params.set("yolo_conf", options.yoloConf.toFixed(2));
  }
  const qs = params.toString() ? `?${params.toString()}` : "";

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const response = await fetch(`${API_BASE_URL}/scan-stream${qs}`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (response.status === 404 || response.status === 405) {
      clearTimeout(timeout);
      return analyzeMedicineImage(fileOrBlob);
    }

    if (!response.ok) {
      let detail = `Scan failed with HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        if (errBody?.detail) detail = String(errBody.detail);
      } catch {
        // ignore JSON parse failure
      }
      throw new Error(detail);
    }

    if (!response.body) {
      clearTimeout(timeout);
      return analyzeMedicineImage(fileOrBlob);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResult: ScanAnalysisResult | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.type === "stage" && onProgress) {
          onProgress({
            stageIndex: Number(msg.stageIndex),
            status: msg.status === "done" ? "done" : "running",
            durationMs:
              typeof msg.durationMs === "number" ? msg.durationMs : undefined,
          });
        } else if (msg.type === "error") {
          throw new Error(String(msg.detail || "Backend scan pipeline failed."));
        } else if (msg.type === "result" && msg.data) {
          finalResult = msg.data as ScanAnalysisResult;
        }
      }
    }

    if (buffer.trim()) {
      const msg = JSON.parse(buffer.trim());
      if (msg.type === "error") {
        throw new Error(String(msg.detail || "Backend scan pipeline failed."));
      }
      if (msg.type === "result" && msg.data) {
        finalResult = msg.data as ScanAnalysisResult;
      }
    }

    if (!finalResult) {
      throw new Error("Scan stream ended without returning results.");
    }

    return { data: finalResult, isLiveBackend: true };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(
        "Scan timed out. Please ensure the FastAPI backend is running and try again."
      );
    }
    throw err instanceof Error
      ? err
      : new Error("Unable to reach the OushudhCheck backend server.");
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Fetches the list of real benchmark medicine strip photos from `test/` via FastAPI.
 */
export async function fetchBenchmarkSamples(): Promise<BenchmarkSample[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/benchmark-samples`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      return (data.samples || []) as BenchmarkSample[];
    }
  } catch {
    // ignore
  }
  return [
    {
      id: "multi-5-strips",
      filename: "newimg (3).jpg",
      title: "5-Strip Tabletop Scan (B-Complex, Napa Extra, Diazepam, Amdocal, Tufnil)",
      titleBn: "৫টি ওষুধের পাতা একসাথে স্ক্যান (ভিটামিন বি-কমপ্লেক্স, নাপা এক্সট্রা, ডায়াজিপাম, অ্যামডোক্যাল, টাফনিল)",
      stripsCount: 5,
      imageUrl: `${API_BASE_URL}/demo-image/multi-5-strips`,
    },
    {
      id: "multi-4-strips",
      filename: "newimg (2).jpg",
      title: "4-Strip Tabletop Scan (Diazepam, Napa Extra, Tufnil, Amdocal)",
      titleBn: "৪টি ওষুধের পাতা স্ক্যান (ডায়াজিপাম, নাপা এক্সট্রা, টাফনিল, অ্যামডোক্যাল)",
      stripsCount: 4,
      imageUrl: `${API_BASE_URL}/demo-image/multi-4-strips`,
    },
    {
      id: "multi-2-emistat-fluver",
      filename: "img1.jpg",
      title: "2-Strip Reflective Foil Scan (Emistat 8mg & Fluver 10mg)",
      titleBn: "২টি ফয়েল স্ট্রিপ স্ক্যান (ইমিস্ট্যাট ৮ মি.গ্রা. ও ফ্লুভার ১০ মি.গ্রা.)",
      stripsCount: 2,
      imageUrl: `${API_BASE_URL}/demo-image/multi-2-emistat-fluver`,
    },
  ];
}

/**
 * Triggers a live 6-stage AI benchmark scan on the FastAPI backend using a real test image from `test/`.
 */
export async function runLiveBenchmarkScan(
  sampleId = "multi-5-strips"
): Promise<{ data: ScanAnalysisResult; isLiveBackend: boolean }> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    const res = await fetch(
      `${API_BASE_URL}/scan-benchmark?sample_id=${encodeURIComponent(sampleId)}`,
      {
        method: "POST",
        signal: controller.signal,
        cache: "no-store",
      }
    );
    clearTimeout(timeout);
    if (res.ok) {
      const json = await res.json();
      return { data: json as ScanAnalysisResult, isLiveBackend: true };
    }
  } catch {
    // Fallback if backend is unreachable
  }

  const preset = PRESET_SCENARIOS[0];
  return {
    data: preset.result,
    isLiveBackend: false,
  };
}
