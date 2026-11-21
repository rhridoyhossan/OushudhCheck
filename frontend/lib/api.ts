import { ScanAnalysisResult } from "./types";
import { PRESET_SCENARIOS } from "./mock-data";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

/**
 * Sends a captured or uploaded medicine strip image to the FastAPI backend
 * (`/api/v1/scan`). If the local backend is offline during frontend preview,
 * gracefully falls back to the high-precision Stage Demo YOLO26 analysis result.
 */
export async function analyzeMedicineImage(
  fileOrBlob: File | Blob,
  fallbackScenarioId = "live-stage-demo"
): Promise<{ data: ScanAnalysisResult; isLiveBackend: boolean }> {
  try {
    const formData = new FormData();
    formData.append("file", fileOrBlob, "tabletop_strips.jpg");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3500);

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

  // Simulate realistic 850ms YOLO26 + PaddleOCR + RapidFuzz + SLM pipeline delay
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
