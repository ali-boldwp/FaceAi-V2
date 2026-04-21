import { AnalyzeResponse } from "../api/client";

type Props = {
  measurements: AnalyzeResponse["measurements"];
  ratios: AnalyzeResponse["ratios"];
};

export function MeasurementsTable({ measurements, ratios }: Props) {
  const idealRatios = ratios.filter((ratio) => ratio.ideal_value !== null && ratio.ideal_value !== undefined);
  const classifiedRatios = ratios.filter((ratio) => ratio.ideal_value === null || ratio.ideal_value === undefined);

  return (
    <div className="card">
      <h2>Measurements</h2>
      <div className="table">
        <div className="row row-measurements header">
          <span>ID</span>
          <span>Label</span>
          <span>Image</span>
          <span>Value (px)</span>
          <span>Notes</span>
        </div>
        {measurements.map((m) => (
          <div key={m.id} className="row row-measurements">
            <span>{m.id}</span>
            <span>{m.label}</span>
            <span>{m.image}</span>
            <span>{m.value !== null ? m.value.toFixed(2) : "-"}</span>
            <span>{m.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Ideal Ratios</h3>
      <div className="table">
        <div className="row row-ratios-ideal header">
          <span>ID</span>
          <span>Label</span>
          <span>Formula</span>
          <span>Value</span>
          <span>Ideal</span>
          <span>Deviation %</span>
          <span>Notes</span>
        </div>
        {idealRatios.map((r) => (
          <div key={r.id} className="row row-ratios-ideal">
            <span>{r.id}</span>
            <span>{r.label || ""}</span>
            <span>{r.numerator} / {r.denominator}</span>
            <span>{r.value !== null ? r.value.toFixed(3) : "-"}</span>
            <span>{r.ideal_value !== null && r.ideal_value !== undefined ? r.ideal_value.toFixed(3) : "-"}</span>
            <span>{r.deviation_pct !== null && r.deviation_pct !== undefined ? r.deviation_pct.toFixed(2) : "-"}</span>
            <span>{r.note || ""}</span>
          </div>
        ))}
      </div>

      <h3>Classifications</h3>
      <div className="table">
        <div className="row row-ratios-classified header">
          <span>ID</span>
          <span>Label</span>
          <span>Formula</span>
          <span>Value</span>
          <span>Interpretation</span>
        </div>
        {classifiedRatios.map((r) => (
          <div key={r.id} className="row row-ratios-classified">
            <span>{r.id}</span>
            <span>{r.label || ""}</span>
            <span>{r.numerator} / {r.denominator}</span>
            <span>{r.value !== null ? r.value.toFixed(3) : "-"}</span>
            <span>{r.note || ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
