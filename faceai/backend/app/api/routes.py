from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from app.models.schemas import AnalyzeResponse, HealthResponse
from app.services.facemesh import analyze_images

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    tr_x: float | None = Form(None),
    tr_y: float | None = Form(None),
    gender: str | None = Form(None),
) -> AnalyzeResponse:
    if front_image.content_type is None or not front_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="front_image must be an image file")
    if side_image.content_type is None or not side_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="side_image must be an image file")
    if tr_x is not None and not (0.0 <= tr_x <= 1.0):
        raise HTTPException(status_code=400, detail="tr_x must be between 0 and 1")
    if tr_y is not None and not (0.0 <= tr_y <= 1.0):
        raise HTTPException(status_code=400, detail="tr_y must be between 0 and 1")
    if gender is not None:
        allowed = {"male", "female", "nonbinary", "prefer_not_to_say"}
        if gender not in allowed:
            raise HTTPException(status_code=400, detail="gender must be a valid option")

    front_bytes = await front_image.read()
    side_bytes = await side_image.read()

    try:
        return analyze_images(front_bytes, side_bytes, tr_x=tr_x, tr_y=tr_y, gender=gender)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
