import { useState } from "react";
import { AnalyzeResponse } from "../api/client";
import { UploadForm } from "../components/UploadForm";
import { ResultsPanel } from "../components/ResultsPanel";

export function Home() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [gender, setGender] = useState("male");
  const [recalcLoading, setRecalcLoading] = useState(false);

  const handleRecalculate = async () => {
    if (!frontFile || !sideFile) {
      setError("Please re-upload front and side images to recalculate.");
      return;
    }
    setRecalcLoading(true);
    setError("");
    try {
      const { analyzeImages } = await import("../api/client");
      const updated = await analyzeImages(frontFile, sideFile, undefined, gender);
      setResult(updated);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to recalculate images.";
      setError(message);
    } finally {
      setRecalcLoading(false);
    }
  };

  return (
    <main className="container">
      <header className="hero">
        <div>
          <p className="eyebrow">FaceAI Geometry Lab</p>
          <h1>Measure facial geometry from front and side images.</h1>
          <p className="subhead">
            Upload two images to receive annotated landmarks, raw measurements, and ratios.
          </p>
        </div>
      </header>

      {!result && (
        <UploadForm
          onResult={setResult}
          onError={setError}
          onFilesChange={(front, side) => {
            setFrontFile(front);
            setSideFile(side);
          }}
          onGenderChange={(value) => setGender(value)}
        />
      )}

      {error && <div className="card error">{error}</div>}

      {result && (
        <>
          <div className="card">
            <button className="primary" type="button" onClick={() => setResult(null)}>
              Analyze another
            </button>
          </div>
          <ResultsPanel
            result={result}
            onRecalculate={handleRecalculate}
            recalculating={recalcLoading}
            onManualTr={async (tr) => {
              if (!frontFile || !sideFile) {
                setError("Please re-upload front and side images to set Tr manually.");
                return;
              }
              try {
                const { analyzeImages } = await import("../api/client");
                const updated = await analyzeImages(frontFile, sideFile, tr, gender);
                setResult(updated);
              } catch (err) {
                const message = err instanceof Error ? err.message : "Failed to apply manual Tr.";
                setError(message);
              }
            }}
          />
        </>
      )}
    </main>
  );
}
