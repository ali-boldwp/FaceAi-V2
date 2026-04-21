# FaceAI

FaceAI is a monorepo for analyzing facial geometry from two images (front and side). It produces annotated images, a mandatory landmark set, and measurement outputs based on configurable landmark mappings.

## Project overview
- Frontend: React + Vite + TypeScript
- Backend: FastAPI + MediaPipe FaceMesh
- Input: front image + side image
- Output: annotated images, JSON landmarks, and measurement/ratio results

## Docker run (primary)
```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## Local dev (secondary)
```bash
npm install
npm run dev
```
This runs the Vite dev server and the FastAPI backend with hot reload.

If you see an error about `mediapipe` missing `solutions`, reinstall backend deps to the pinned version:
```bash
pip install -r backend/requirements.txt
```

## API usage
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "front_image=@/path/to/front.jpg" \
  -F "side_image=@/path/to/side.jpg"
```

### Response shape
- `annotated_images.front` and `annotated_images.side` are base64 PNGs with the `data:image/png;base64` prefix.
- `mandatory_landmarks` includes pixel + normalized coordinates when available.
- `measurements` includes `value` in pixels or `null` with a note when missing.

## Landmark mapping guide
Landmark indices are stored in `backend/app/utils/landmarks_map.json`. Update this file to refine which MediaPipe FaceMesh indices correspond to each anthropometric label. Any `null` values will be skipped from required measurements.

## Known limitations
- 2D measurements from single images are sensitive to lighting and head pose.
- Side-image depth measurements are approximate without calibrated cameras.
- Landmark index mappings require domain-specific validation.

## Hairline (Tr) estimation
Trichion is estimated using a BiSeNet hair segmentation model downloaded on first run. The model is cached under
`model_cache/face_parsing` at the repo root. If the model cannot be downloaded or hair is not detected, Tr-based measurements
will be returned as `null` with a warning.

## Safety
The system uses geometry-only outputs and does not attempt any personality or temperament inference. Uploaded images are processed in memory and not stored.
