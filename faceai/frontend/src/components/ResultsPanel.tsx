import { useState } from "react";
import { AnalyzeResponse } from "../api/client";
import { LandmarksViewer } from "./LandmarksViewer";
import { MeasurementsTable } from "./MeasurementsTable";

type Props = {
  result: AnalyzeResponse;
  onRecalculate: () => void;
  recalculating: boolean;
  onManualTr: (tr: { x: number; y: number }) => void;
};

export function ResultsPanel({ result, onRecalculate, recalculating, onManualTr }: Props) {
  const [showAllLandmarks, setShowAllLandmarks] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [zoomSrc, setZoomSrc] = useState<string | null>(null);
  const [zoomAlt, setZoomAlt] = useState<string>("");
  const [manualTrMode, setManualTrMode] = useState(false);

  const openZoom = (src: string, alt: string) => {
    setZoomSrc(src);
    setZoomAlt(alt);
  };

  const closeZoom = () => {
    setZoomSrc(null);
    setZoomAlt("");
  };

  const shouldOfferManualTr = result.warnings.some((warning) =>
    warning.toLowerCase().includes("trichion")
  );

  const handleFrontClick = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!manualTrMode) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    onManualTr({ x, y });
    setManualTrMode(false);
  };

  return (
    <section className="results">
      <div className="card">
        <h2>Annotated Images</h2>
        <div className="manual-tr">
          <button className="primary" type="button" onClick={onRecalculate} disabled={recalculating}>
            {recalculating ? "Recalculating..." : "Recalculate images"}
          </button>
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showAllLandmarks}
            onChange={(e) => setShowAllLandmarks(e.target.checked)}
          />
          Show all landmarks (indices)
        </label>
        {shouldOfferManualTr && (
          <div className="manual-tr">
            <button
              className="primary"
              type="button"
              onClick={() => setManualTrMode((prev) => !prev)}
            >
              {manualTrMode ? "Cancel Tr selection" : "Set Tr manually"}
            </button>
            <p className="hint">
              {manualTrMode
                ? "Click the Tr point (hairline/forehead start) on the front image."
                : "If hairline is not detected, you can set Tr manually."}
            </p>
          </div>
        )}
        <div className={`image-grid ${manualTrMode ? "selecting" : ""}`}>
          <figure>
            <img
              src={result.annotated_images.front}
              alt="Annotated front"
              className="zoomable"
              onClick={(event) =>
                manualTrMode
                  ? handleFrontClick(event)
                  : openZoom(result.annotated_images.front, "Annotated front")
              }
            />
            <figcaption>Front</figcaption>
          </figure>
          <figure>
            <img
              src={result.annotated_images.side}
              alt="Annotated side"
              className="zoomable"
              onClick={() => openZoom(result.annotated_images.side, "Annotated side")}
            />
            <figcaption>Side</figcaption>
          </figure>
          {showAllLandmarks && (
            <>
              <figure>
                <img
                  src={result.annotated_images.front_all}
                  alt="Front all landmarks"
                  className="zoomable"
                  onClick={() => openZoom(result.annotated_images.front_all, "Front all landmarks")}
                />
                <figcaption>Front (all landmarks)</figcaption>
              </figure>
              <figure>
                <img
                  src={result.annotated_images.side_all}
                  alt="Side all landmarks"
                  className="zoomable"
                  onClick={() => openZoom(result.annotated_images.side_all, "Side all landmarks")}
                />
                <figcaption>Side (all landmarks)</figcaption>
              </figure>
            </>
          )}
        </div>
      </div>

      {Object.keys(result.annotated_images).some((key) => key.startsWith("tr_")) && (
        <div className="card">
          <h2>Hairline Debug</h2>
          <label className="toggle">
            <input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />
            Show hairline debug images
          </label>
          {showDebug && (
            <div className="image-grid">
              {Object.entries(result.annotated_images)
                .filter(([key, value]) => key.startsWith("tr_") && value)
                .map(([key, value]) => (
                  <figure key={key}>
                    <img src={value} alt={key} className="zoomable" onClick={() => openZoom(value, key)} />
                    <figcaption>{key}</figcaption>
                  </figure>
                ))}
            </div>
          )}
        </div>
      )}

      <MeasurementsTable measurements={result.measurements} ratios={result.ratios} />
      <LandmarksViewer landmarks={result.mandatory_landmarks} warnings={result.warnings} />

      {zoomSrc && (
        <div className="zoom-overlay" role="dialog" aria-modal="true" onClick={closeZoom}>
          <div className="zoom-content" onClick={(e) => e.stopPropagation()}>
            <button className="zoom-close" type="button" onClick={closeZoom}>
              Close
            </button>
            <img src={zoomSrc} alt={zoomAlt} />
          </div>
        </div>
      )}
    </section>
  );
}
