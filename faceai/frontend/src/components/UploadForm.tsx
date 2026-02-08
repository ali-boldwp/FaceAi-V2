import { useState } from "react";
import { analyzeImages, AnalyzeResponse } from "../api/client";

type Props = {
  onResult: (result: AnalyzeResponse) => void;
  onError: (message: string) => void;
  onFilesChange: (front: File | null, side: File | null) => void;
  onGenderChange: (gender: string) => void;
};

export function UploadForm({ onResult, onError, onFilesChange, onGenderChange }: Props) {
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [gender, setGender] = useState("male");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!frontFile || !sideFile) {
      onError("Please select both a front and side image.");
      return;
    }

    setLoading(true);
    onError("");

    try {
      const result = await analyzeImages(frontFile, sideFile, undefined, gender);
      onResult(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unexpected error";
      onError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Upload Images</h2>
      <div className="grid">
        <label className="field">
          <span>Front Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              setFrontFile(file);
              onFilesChange(file, sideFile);
            }}
          />
        </label>
        <label className="field">
          <span>Side Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              setSideFile(file);
              onFilesChange(frontFile, file);
            }}
          />
        </label>
        <label className="field">
          <span>Gender (Self-Reported)</span>
          <select
            value={gender}
            onChange={(e) => {
              setGender(e.target.value);
              onGenderChange(e.target.value);
            }}
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="nonbinary">Non-binary</option>
            <option value="prefer_not_to_say">Prefer not to say</option>
          </select>
        </label>
      </div>
      <button className="primary" type="submit" disabled={loading}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>
      <p className="hint">
        Images are processed in memory and are not stored.
      </p>
    </form>
  );
}
