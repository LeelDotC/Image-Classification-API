"""FastAPI routes with one shared CPU classifier and validated image uploads."""

from contextlib import asynccontextmanager, closing
from io import BytesIO
import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.model_service import ImageClassifier, MODEL_ID
from app.schemas import APIInfo, HealthResponse, PredictionResponse


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 5_242_880
SUPPORTED_FORMATS = ("JPEG", "PNG")


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.classifier = None
    try:
        application.state.classifier = await run_in_threadpool(ImageClassifier)
    except Exception:
        logger.exception("Classifier initialization failed; application startup aborted")
        raise
    logger.info("Classifier initialized for %s", MODEL_ID)
    try:
        yield
    finally:
        application.state.classifier = None


app = FastAPI(title="Image Classification API", lifespan=lifespan)


@app.get("/", response_model=APIInfo)
async def information() -> APIInfo:
    return APIInfo(
        name="Image Classification API",
        model=MODEL_ID,
        supported_formats=list(SUPPORTED_FORMATS),
    )


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health(request: Request, response: Response) -> HealthResponse:
    loaded = getattr(request.app.state, "classifier", None) is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(status="ok" if loaded else "unavailable", model_loaded=loaded)


def _classify_image(classifier: ImageClassifier, data: bytes) -> dict:
    """Decode and classify off the event loop, closing all Pillow resources."""
    with BytesIO(data) as stream:
        try:
            with BytesIO(data) as verification_stream, closing(Image.open(verification_stream)) as header:
                if header.format not in SUPPORTED_FORMATS:
                    raise HTTPException(415, "Only JPEG and PNG images are supported")
                if header.format == "PNG" and getattr(header, "is_animated", False):
                    raise HTTPException(400, "Animated PNG images are not supported")
                header.verify()
            image = Image.open(stream)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Upload is not a valid image") from exc

        with closing(image):
            try:
                image.load()
            except (OSError, ValueError, SyntaxError, EOFError) as exc:
                raise HTTPException(400, "Image could not be fully decoded") from exc
            return classifier.classify(image)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"description": "Empty, corrupt, or animated image"},
        413: {"description": "File exceeds 5 MiB"},
        415: {"description": "Unsupported decoded image format"},
        500: {"description": "Image classification failed"},
        503: {"description": "Classifier is unavailable"},
    },
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="One JPEG or PNG image, at most 5 MiB")],
) -> PredictionResponse:
    form = await request.form()
    try:
        uploads = [value for _, value in form.multi_items() if isinstance(value, StarletteUploadFile)]
        if len(uploads) != 1:
            raise HTTPException(422, "Provide exactly one uploaded image in field 'file'")

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "Uploaded image is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 5 MiB (5,242,880 byte) limit")

        classifier = getattr(request.app.state, "classifier", None)
        if classifier is None:
            raise HTTPException(503, "Classifier is unavailable")
        try:
            result = await run_in_threadpool(_classify_image, classifier, data)
            return PredictionResponse(model=MODEL_ID, **result)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Image classification failed")
            raise HTTPException(500, "Image classification failed") from None
    finally:
        await form.close()
