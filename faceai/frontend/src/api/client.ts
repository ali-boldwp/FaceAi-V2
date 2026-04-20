const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const INVALID_SIDE_IMAGE_MESSAGE =
  "Imaginea laterală nu a fost detectată corect. Te rugăm să încarci o fotografie la 45-60 de grade, cu nasul, un ochi, urechea, linia maxilarului și bărbia vizibile clar.";

export class SideImageDetectionError extends Error {
  imageSrc = "/side-reference.png";

  constructor() {
    super(INVALID_SIDE_IMAGE_MESSAGE);
    this.name = "SideImageDetectionError";
  }
}

export type AnalyzeResponse = {
  ok: boolean;
  all_landmarks_count: number;
  gender: string | null;
  mandatory_landmarks: Array<{
    label: string;
    index: number | null;
    pixel: { x: number; y: number } | null;
    normalized: { x: number; y: number; z: number } | null;
  }>;
  measurements: Array<{
    id: string;
    label: string;
    image: string;
    points: string[];
    value: number | null;
    unit: string;
    note?: string | null;
  }>;
  ratios: Array<{
    id: string;
    numerator: string;
    denominator: string;
    value: number | null;
    note?: string | null;
  }>;
  annotated_images: {
    front: string;
    side: string;
    front_all: string;
    side_all: string;
    tr_hair_mask?: string;
    tr_overlay?: string;
    [key: string]: string | undefined;
  };
  warnings: string[];
};

export async function analyzeImages(
  front: File,
  side: File,
  trOverride?: { x: number; y: number },
  gender?: string
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("front_image", front);
  form.append("side_image", side);
  if (trOverride) {
    form.append("tr_x", String(trOverride.x));
    form.append("tr_y", String(trOverride.y));
  }
  if (gender) {
    form.append("gender", gender);
  }

  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Analysis failed");
  }

  const data = (await res.json()) as AnalyzeResponse;
  const sideMissing = data.warnings.some((warning) =>
    warning.toLowerCase().includes("no face detected in side image")
  );
  if (sideMissing) {
    throw new SideImageDetectionError();
  }

  return data;
}
