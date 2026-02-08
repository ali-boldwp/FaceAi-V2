import { AnalyzeResponse } from "../api/client";

type Props = {
  landmarks: AnalyzeResponse["mandatory_landmarks"];
  warnings: string[];
};

export function LandmarksViewer({ landmarks, warnings }: Props) {
  return (
    <div className="card">
      <h2>Landmarks</h2>
      {warnings.length > 0 && (
        <div className="warnings">
          {warnings.map((warning, index) => (
            <p key={index}>{warning}</p>
          ))}
        </div>
      )}
      <details>
        <summary>View landmark JSON</summary>
        <pre>{JSON.stringify(landmarks, null, 2)}</pre>
      </details>
    </div>
  );
}
