import { AnalyzeResponse } from "../api/client";

type Props = {
  measurements: AnalyzeResponse["measurements"];
  ratios: AnalyzeResponse["ratios"];
  idealFace: AnalyzeResponse["ideal_face"];
  eyes: AnalyzeResponse["eyes"];
  forehead?: AnalyzeResponse["forehead"];
  nose?: AnalyzeResponse["nose"];
  mouth?: AnalyzeResponse["mouth"];
  jaw?: AnalyzeResponse["jaw"];
  cheek?: AnalyzeResponse["cheek"];
  eyebrows?: AnalyzeResponse["eyebrows"];
  ears?: AnalyzeResponse["ears"];
};

function formatValue(value: number | null | undefined, digits = 3) {
  return value !== null && value !== undefined ? value.toFixed(digits) : "-";
}

function metricValue(value: number | null, unit: string) {
  if (value === null) return "-";
  if (unit === "deg") return `${value.toFixed(2)} deg`;
  if (unit === "percent") return `${value.toFixed(2)}%`;
  if (unit === "px") return `${value.toFixed(2)} px`;
  return value.toFixed(3);
}

export function MeasurementsTable({ measurements, ratios, idealFace, eyes, forehead, nose, mouth, jaw, cheek, eyebrows, ears }: Props) {
  return (
    <div className="card">
      <h2>Measurements</h2>
      <div className="table">
        <div className="row header">
          <span>ID</span>
          <span>Label</span>
          <span>Image</span>
          <span>Value (px)</span>
          <span>Notes</span>
        </div>
        {measurements.map((m) => (
          <div key={m.id} className="row">
            <span>{m.id}</span>
            <span>{m.label}</span>
            <span>{m.image}</span>
            <span>{m.value !== null ? m.value.toFixed(2) : "-"}</span>
            <span>{m.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Ratios</h3>
      <div className="table">
        <div className="row header">
          <span>ID</span>
          <span>Formula</span>
          <span>Value</span>
          <span>Notes</span>
        </div>
        {ratios.map((r) => (
          <div key={r.id} className="row">
            <span>{r.id}</span>
            <span>{r.numerator} / {r.denominator}</span>
            <span>{formatValue(r.value)}</span>
            <span>{r.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Ideal Face (Phi = {idealFace.phi})</h3>
      <p className="hint">
        Base reference Wₒ (eye width): {formatValue(idealFace.base_eye_width, 2)} px
      </p>
      <div className="table">
        <div className="row header row-ideal-ratios">
          <span>Ratio</span>
          <span>Formula</span>
          <span>Actual</span>
          <span>Ideal</span>
          <span>Diff. %</span>
          <span>Interpretation</span>
        </div>
        {idealFace.ratios.map((ratio) => (
          <div key={ratio.id} className="row row-ideal-ratios">
            <span>{ratio.label}</span>
            <span>{ratio.formula}</span>
            <span>{formatValue(ratio.actual)}</span>
            <span>{ratio.ideal.toFixed(3)}</span>
            <span>{formatValue(ratio.diff_percent, 1)}</span>
            <span>{ratio.interpretation || ratio.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Wₒ Grid (ideal units)</h3>
      <div className="table">
        <div className="row header row-ideal-dimensions">
          <span>Landmark</span>
          <span>Measured</span>
          <span>Ideal</span>
          <span>Note</span>
        </div>
        {idealFace.dimensions.map((dimension) => (
          <div key={dimension.id} className="row row-ideal-dimensions">
            <span>{dimension.label}</span>
            <span>{formatValue(dimension.actual, 2)} {dimension.unit}</span>
            <span>{formatValue(dimension.ideal, 2)} {dimension.unit}</span>
            <span>{dimension.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Face Shape Classification (10 types)</h3>
      <div className="table">
        <div className="row header row-face-indices">
          <span>IFV (H/W)</span>
          <span>IZG (W/J)</span>
          <span>IFM (F/J)</span>
          <span>IFZF (W/F)</span>
          <span>Shape</span>
        </div>
        <div className="row row-face-indices">
          <span>{formatValue(idealFace.indices.ifv)}</span>
          <span>{formatValue(idealFace.indices.izg)}</span>
          <span>{formatValue(idealFace.indices.ifm)}</span>
          <span>{formatValue(idealFace.indices.ifzf)}</span>
          <span>
            {idealFace.face_shape || "-"}
            {idealFace.face_shape_note ? ` (${idealFace.face_shape_note})` : ""}
          </span>
        </div>
      </div>

      <h3>Eyes Atlas (Objective)</h3>
      <p className="hint">
        Zoomorphic label: <strong>{eyes.zoomorphic_label || "-"}</strong>
        {eyes.zoomorphic_note ? ` - ${eyes.zoomorphic_note}` : ""}
      </p>
      <p className="hint">Standard IDs: {eyes.standard_ids.join(", ")}</p>
      <p className="hint">Output signature: {eyes.output_signature || "-"}</p>
      <div className="table">
        <div className="row header row-eyes-metrics">
          <span>ID</span>
          <span>Metric</span>
          <span>Value</span>
          <span>Class</span>
          <span>Notes</span>
        </div>
        {eyes.metrics.map((metric) => (
          <div key={metric.id} className="row row-eyes-metrics">
            <span>{metric.id}</span>
            <span>{metric.label}</span>
            <span>{metricValue(metric.value, metric.unit)}</span>
            <span>{metric.classification || "-"}</span>
            <span>{metric.note || ""}</span>
          </div>
        ))}
      </div>
      <div className="table">
        <div className="row header row-eye-classes">
          <span>Form</span>
          <span>Size</span>
          <span>Orientation</span>
          <span>Spacing</span>
          <span>Symmetry</span>
        </div>
        <div className="row row-eye-classes">
          <span>{eyes.classification.form || "-"}</span>
          <span>{eyes.classification.size || "-"}</span>
          <span>{eyes.classification.orientation || "-"}</span>
          <span>{eyes.classification.spacing || "-"}</span>
          <span>{eyes.classification.symmetry || "-"}</span>
        </div>
      </div>

      {forehead && (
        <>
          <h3>Forehead Analysis</h3>
          <p className="hint">Signature: {forehead.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Height</span>
              <span>Width</span>
            </div>
            <div className="row">
              <span>{forehead.height_classification || "-"}</span>
              <span>{forehead.width_classification || "-"}</span>
            </div>
          </div>
        </>
      )}

      {nose && (
        <>
          <h3>Nose Analysis</h3>
          <p className="hint">Signature: {nose.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Width</span>
              <span>Projection</span>
              <span>Base</span>
            </div>
            <div className="row">
              <span>{nose.width_classification || "-"}</span>
              <span>{nose.projection_classification || "-"}</span>
              <span>{nose.base_classification || "-"}</span>
            </div>
          </div>
        </>
      )}

      {mouth && (
        <>
          <h3>Mouth Analysis</h3>
          <p className="hint">Signature: {mouth.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Width</span>
              <span>Volume</span>
              <span>Ratio</span>
            </div>
            <div className="row">
              <span>{mouth.width_classification || "-"}</span>
              <span>{mouth.volume_classification || "-"}</span>
              <span>{mouth.ratio_classification || "-"}</span>
            </div>
          </div>
        </>
      )}

      {jaw && (
        <>
          <h3>Jaw & Chin Analysis</h3>
          <p className="hint">Signature: {jaw.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Mandible Type</span>
            </div>
            <div className="row">
              <span>{jaw.mandible_type || "-"}</span>
            </div>
          </div>
        </>
      )}

      {cheek && (
        <>
          <h3>Cheeks & Cheekbones Analysis</h3>
          <p className="hint">Signature: {cheek.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Bone Class</span>
            </div>
            <div className="row">
              <span>{cheek.bone_classification || "-"}</span>
            </div>
          </div>
        </>
      )}

      {ears && (
        <>
          <h3>Ears Analysis</h3>
          <p className="hint">Signature: {ears.output_signature || "-"}</p>
          <div className="table">
            <div className="row header">
              <span>Form</span>
            </div>
            <div className="row">
              <span>{ears.form_classification || "-"}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
