import { BenchmarkSample, ScanAnalysisResult } from "./types";
import { PRESET_SCENARIOS } from "./mock-data";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export const BACKEND_ORIGIN = API_BASE_URL.replace(/\/api\/v1\/?$/, "");

/**
 * Sends a captured or uploaded medicine strip image to the FastAPI backend
 * (`/api/v1/scan`).
 */
export async function analyzeMedicineImage(
  fileOrBlob: File | Blob,
  fallbackScenarioId = "live-stage-demo"
): Promise<{ data: ScanAnalysisResult; isLiveBackend: boolean }> {
  try {
    const formData = new FormData();
    const fileName =
      fileOrBlob instanceof File && fileOrBlob.name
        ? fileOrBlob.name
        : "tabletop_strips.jpg";
    formData.append("file", fileOrBlob, fileName);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);

    const response = await fetch(`${API_BASE_URL}/scan`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (response.ok) {
      const json = await response.json();
      return { data: json as ScanAnalysisResult, isLiveBackend: true };
    }
  } catch {
    // Backend not running locally; fallback to deterministic demo simulation
  }

  await new Promise((resolve) => setTimeout(resolve, 850));
  const preset =
    PRESET_SCENARIOS.find((s) => s.id === fallbackScenarioId) ||
    PRESET_SCENARIOS[0];

  return {
    data: {
      ...preset.result,
      timestamp: `Scanned at ${new Date().toLocaleTimeString()} • Edge ONNX Runtime`,
    },
    isLiveBackend: false,
  };
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
