import { AnalyzeResponse } from "../api/client";

type Props = {
  measurements: AnalyzeResponse["measurements"];
  ratios: AnalyzeResponse["ratios"];
};

export function MeasurementsTable({ measurements, ratios }: Props) {
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
            <span>{r.value !== null ? r.value.toFixed(3) : "-"}</span>
            <span>{r.note || ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
