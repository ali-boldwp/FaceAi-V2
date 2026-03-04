const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
  ideal_face: {
    phi: number;
    base_eye_width: number | null;
    ratios: Array<{
      id: string;
      label: string;
      formula: string;
      actual: number | null;
      ideal: number;
      diff_percent: number | null;
      deviation: string | null;
      interpretation: string | null;
      note?: string | null;
    }>;
    dimensions: Array<{
      id: string;
      label: string;
      actual: number | null;
      ideal: number | null;
      unit: string;
      note?: string | null;
    }>;
    indices: {
      ifv: number | null;
      izg: number | null;
      ifm: number | null;
      ifzf: number | null;
    };
    face_shape: string | null;
    face_shape_note: string | null;
  };
  eyes: {
    standard_ids: string[];
    metrics: Array<{
      id: string;
      label: string;
      value: number | null;
      unit: string;
      classification: string | null;
      note?: string | null;
    }>;
    composite_indices: Record<string, number | null>;
    classification: {
      form: string | null;
      size: string | null;
      orientation: string | null;
      spacing: string | null;
      symmetry: string | null;
      eyelid: string | null;
      depth: string | null;
    };
    zoomorphic_label: string | null;
    zoomorphic_note: string | null;
    output_signature: string | null;
  };
  forehead?: {
    r1: number | null;
    r2: number | null;
    r3: number | null;
    r4: number | null;
    f7: number | null;
    height_classification: string | null;
    width_classification: string | null;
    profile_classification: string | null;
    relief_classification: string | null;
    symmetry_classification: string | null;
    output_signature: string | null;
  };
  nose?: {
    in_index: number | null;
    ip_index: number | null;
    ib_index: number | null;
    il_index: number | null;
    nla_angle: number | null;
    nfa_angle: number | null;
    width_classification: string | null;
    projection_classification: string | null;
    base_classification: string | null;
    rotation_classification: string | null;
    symmetry_classification: string | null;
    output_signature: string | null;
  };
  mouth?: {
    imw: number | null;
    igb: number | null;
    itb: number | null;
    ivv: number | null;
    width_classification: string | null;
    volume_classification: string | null;
    ratio_classification: string | null;
    symmetry_classification: string | null;
    output_signature: string | null;
  };
  jaw?: {
    r1: number | null;
    r3: number | null;
    r5: number | null;
    c6_angle: number | null;
    i1: number | null;
    i2: number | null;
    i3: number | null;
    jp1_angle: number | null;
    jm4_angle: number | null;
    mandible_type: string | null;
    chin_type: string | null;
    profile_type: string | null;
    coherence_flag: string | null;
    output_signature: string | null;
  };
  cheek?: {
    rz1: number | null;
    rz2: number | null;
    rz3: number | null;
    ro1: number | null;
    ro2: number | null;
    ro3: number | null;
    bone_classification: string | null;
    volume_classification: string | null;
    output_signature: string | null;
  };
  eyebrows?: {
    bed: number | null;
    bt: number | null;
    delta_h: number | null;
    bta_angle: number | null;
    bl_ratio: number | null;
    form_classification: string | null;
    position_classification: string | null;
    thickness_classification: string | null;
    length_classification: string | null;
    tail_classification: string | null;
    symmetry_classification: string | null;
    output_signature: string | null;
  };
  ears?: {
    el: number | null;
    ew: number | null;
    ll: number | null;
    ed: number | null;
    ie: number | null;
    il_index: number | null;
    ip: number | null;
    length_classification: string | null;
    form_classification: string | null;
    lob_classification: string | null;
    protrusion_classification: string | null;
    symmetry_classification: string | null;
    output_signature: string | null;
  };
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

  return res.json();
}
