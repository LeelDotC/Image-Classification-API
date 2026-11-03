# Image Classification API

## Student information
- Name: Lê Tấn Lộc
- Student ID: 25127406
- Class: 25C09 

## Project description

This project uses a pretrained image-classification model to predict the main subject of a photograph. It provides a FastAPI HTTP API and a separate command-line runner. No custom model training is performed.

The workflow is:

**Upload an image → validate the file → preprocess pixels → classify on CPU → return JSON.**

The API checks the uploaded file, converts grayscale images to RGB and composites transparent images onto white, then uses the matching Hugging Face processor for resizing, rescaling, and normalization. The model returns scores for its fixed categories; the application returns the three highest-scoring labels.

The assignment requires a pretrained Hugging Face model exposed through FastAPI, information and health endpoints, and a prediction endpoint chosen from `/predict` or `/generate`. This project implements `/predict`. The `file` field, JPEG/PNG restriction, 5 MiB limit, and top-three response are project design choices.

## Pretrained model

- **Identifier:** `google/vit-base-patch16-224`, verified in `app/model_service.py`.
- **Hugging Face model card:** [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224).
- **Architecture:** base-sized Vision Transformer (ViT), using 16 × 16 image patches.
- **Training background:** pretrained on ImageNet-21k and fine-tuned on ImageNet-1k; the model card lists Apache-2.0.
- **Output vocabulary:** the **1,000 ImageNet-1k classes**, not arbitrary labels or all pretraining categories. The full set is the checkpoint's [`id2label` configuration](https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json).

The categories include animals, dog breeds, vehicles, household objects, foods, and other ImageNet subjects. Labels retain their complete configuration strings: for example, `sports car, sport car` is one label with synonyms, not two separate predictions.

The implementation loads `ViTForImageClassification` and `ViTImageProcessorPil` once per application process during lifespan startup. It runs on CPU in evaluation mode under `torch.inference_mode()`. Importing the module does not load weights. If initialization fails, startup fails; shutdown releases the stored classifier reference. Decoding and inference run in a worker thread rather than blocking the async event loop.

The matching processor resizes to 224 × 224, rescales 8-bit pixels to [0, 1], and normalizes RGB channels with mean 0.5 and standard deviation 0.5. The application does not duplicate those transformations.

Scores are softmax values over all 1,000 classes. The top three are returned in descending order without renormalization, so their scores **need not sum to one**. A score is not a guarantee of correctness or a calibrated real-world confidence estimate.

## Installation with Windows PowerShell

The verified environment is **CPython 3.13.15, Windows x64**, using CPU-only PyTorch. A GPU or CUDA installation is not required. The exact tested direct and transitive dependency versions are pinned in `requirements.txt`.

| Main dependency | Verified version |
| --- | --- |
| fastapi | 0.162.0 |
| uvicorn | 0.54.0 |
| transformers | 5.22.1 |
| torch | 2.15.0+cpu |
| Pillow | 13.0.0 |
| python-multipart | 0.0.33 |
| requests | 2.34.2 |

Open PowerShell and enter the project directory:

```powershell
Set-Location 'C:\Leel.c\HCMUS\26-27\1\Computational_Thinking\LAB\02\Image-Classification-API'
```

For a fresh checkout without `.venv`, create the environment with the installed standard Windows Python interpreter. This is the verified interpreter location on the development machine; adjust it to your own Python installation if necessary:

```powershell
& 'C:\Users\letan\AppData\Local\Programs\Python\Python313\python.exe' -m venv .venv
```

If the project's working `.venv` already exists, skip creation. Install the pinned dependencies into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Activation and PowerShell execution-policy changes are unnecessary. Use the explicit `.venv` interpreter, because the development machine's plain `python` command points to a different MSYS2 installation. The requirements file includes the official PyTorch CPU wheel index. These pins were verified on Windows x64; compatibility on other systems has not been established by this project.

## Run the application

From the project root, start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete` before sending requests. The first startup downloads the pretrained model and processor and **requires Internet access**. Files are cached in `.cache/huggingface/` inside the project, independently of the working directory. Later startups reuse those files while loading the model into memory. The cache is excluded from Git.

If the full checkpoint is already cached, optional offline operation can be enabled for the current terminal:

```powershell
$env:HF_HUB_OFFLINE = '1'
```

Remove that setting when online access is needed again:

```powershell
Remove-Item Env:HF_HUB_OFFLINE
```

Local URLs:

| Address | Purpose |
| --- | --- |
| [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | API information |
| [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | Model readiness |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive Swagger UI |
| [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) | Generated OpenAPI schema |

`127.0.0.1` means the machine running the application. A browser on another computer using this address connects to that other computer, not to this server. The command above binds locally. If port 8000 is occupied, use an available port such as `--port 8001` and update the client URLs. Stop only your own server with Ctrl+C.

## API reference

### GET /

Returns project information. Send a GET request with **no request body or upload**.

Actual HTTP 200 response:

```json
{
  "name": "Image Classification API",
  "model": "google/vit-base-patch16-224",
  "supported_formats": ["JPEG", "PNG"]
}
```

### GET /health

Checks application state without running inference. Send a GET request with **no request body**.

Actual HTTP 200 response from the initialized application:

```json
{"status": "ok", "model_loaded": true}
```

If reachable without an initialized classifier, the code returns HTTP 503 with `{"status":"unavailable","model_loaded":false}`. Normal model-initialization failures prevent application startup instead of reporting a healthy service.

### POST /predict

Accepts **multipart/form-data** containing exactly one uploaded image in the required field **`file`**. Use POST to send image data. Opening `/predict` in the browser address bar sends GET and returns HTTP 405; it does not run a prediction.

| Restriction | Implemented behavior |
| --- | --- |
| Image formats | Genuine JPEG or PNG, determined from decoded content |
| Upload count | Exactly one file; duplicate `file` fields and extra files under other names are rejected |
| File-size limit | **5 MiB = 5,242,880 bytes, inclusive**, excluding multipart-envelope bytes |
| File validity | Nonempty data, Pillow integrity verification, and full decoding before classification |
| Animation | Animated PNGs are rejected |
| Preprocessing | Grayscale becomes RGB; transparency is composited onto white; checkpoint processor handles the remaining transforms |

Extensions and declared MIME types alone do not establish validity. Original images do not need to be 224 × 224. Upload streams, decoded images, and temporary preprocessing images are closed on success and failure; the reusable classifier preserves its caller's image data.

A successful response is HTTP 200 with `application/json`:

- `model`: the Hugging Face model identifier.
- `predictions`: exactly three objects, ordered by descending score.
- Each object contains a nonempty `label` and a finite `score` in [0, 1].

The endpoint uses Pydantic response models. Its errors use a JSON `detail` field; FastAPI request-validation errors can contain a structured list.

| HTTP status | Meaning |
| --- | --- |
| 400 | Empty, invalid, corrupt, truncated, or animated PNG image; malformed multipart requests can also be rejected by the parser |
| 413 | Uploaded file exceeds 5,242,880 bytes |
| 415 | Detected image format is neither JPEG nor PNG, such as a GIF named `.jpg` |
| 422 | Missing/wrong upload field, duplicate files, or additional uploaded files |
| 500 | Unexpected inference or prediction-response failure; technical details are logged server-side |
| 503 | Prediction classifier is unavailable |

For example, an empty file returns HTTP 400 with `{"detail":"Uploaded image is empty"}`. Oversized input returns HTTP 413 with `{"detail":"Image exceeds the 5 MiB (5,242,880 byte) limit"}`. Inference failures return the generic `{"detail":"Image classification failed"}` without exposing internal paths or tracebacks.

## Upload examples

### Python requests

Start the server first. Save this optional example as `upload_example.py` in the project root and run it with `.\.venv\Scripts\python.exe upload_example.py`. This example file is not included in the repository.

```python
import json
from pathlib import Path
import sys

import requests

try:
    with requests.Session() as session:
        with Path("examples/dog.jpg").open("rb") as image:
            response = session.post(
                "http://127.0.0.1:8000/predict",
                files={"file": ("dog.jpg", image, "image/jpeg")},
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()
except requests.HTTPError as error:
    print(f"HTTP {error.response.status_code}: {error.response.text[:300]}", file=sys.stderr)
    raise SystemExit(1)
except (requests.RequestException, OSError, ValueError) as error:
    print(f"Upload failed: {error}", file=sys.stderr)
    raise SystemExit(1)

print(json.dumps(result, indent=2))
```

The example closes its session and image handle, sets a CPU-appropriate timeout, handles unsuccessful HTTP responses, and reports transport, file, or JSON errors. Let `requests` generate the multipart `Content-Type` header, including its boundary.

### Upload through /docs

- Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- Expand **POST /predict** and select **Try it out**.
- Choose `examples/dog.jpg` or another supported image in the **file** input.
- Select **Execute** and wait for inference to finish.
- Read the HTTP status and response body under **Server response**.

Swagger's **Example Value** is generated from the schema and is not a prediction produced by your image. A real result appears under **Server response** only after executing a request. GET `/` and GET `/health` can be executed without a file.

## Actual prediction responses

These responses were obtained from the running local API on **2026-11-03** using the included sample files and the verified CPU environment. They are actual outputs, not Swagger Example Values. Small floating-point differences can occur on other systems.

`examples/dog.jpg`, HTTP 200:

```json
{
  "model": "google/vit-base-patch16-224",
  "predictions": [
    {"label": "Chihuahua", "score": 0.9725014567375183},
    {"label": "toy terrier", "score": 0.006107835099101067},
    {"label": "Pembroke, Pembroke Welsh corgi", "score": 0.004142644815146923}
  ]
}
```

`examples/vehicle.jpg`, HTTP 200:

```json
{
  "model": "google/vit-base-patch16-224",
  "predictions": [
    {"label": "sports car, sport car", "score": 0.9655947089195251},
    {"label": "racer, race car, racing car", "score": 0.017829174175858498},
    {"label": "car wheel", "score": 0.009332562796771526}
  ]
}
```

## Testing

The server **must already be running**. Start it in one PowerShell terminal using the Uvicorn command above. In another terminal at the project root, run:

```powershell
.\.venv\Scripts\python.exe test_api.py
```

The default URL is `http://127.0.0.1:8000`. For a server actually running on another port, use:

```powershell
.\.venv\Scripts\python.exe test_api.py --base-url http://127.0.0.1:8001 --timeout 180
```

`--timeout` is a positive finite per-request connection/read timeout in seconds, defaulting to 120. The script resolves its required sample paths relative to `test_api.py`, rather than the caller's working directory. It does not start/stop servers, load model weights, or mutate server state.

The ten implemented checks are:

- API information and initial health readiness.
- Successful dog and vehicle JPEG predictions, including JSON content type, model identifier, three nonempty labels, finite non-boolean scores in [0, 1], and descending order.
- Missing file (422), text named `invalid.jpg` (400), empty upload (400), and an in-memory file one byte over the limit (413), with meaningful JSON error details.
- A valid prediction after those invalid uploads and a final ready health response.

Each check prints **PASS** or **FAIL**; successful prediction checks also print the actual leading label and score. The final summary includes passed, failed, and unexecuted counts. Exit code **0** means all ten checks passed; **1** means checks failed, required samples were missing, or the server was unreachable. Invalid CLI arguments produce argparse's nonzero usage-error exit. Connection failures stop further HTTP checks; response timeouts fail their checks while independent checks continue. No unexecuted check is marked as passing.

The client does not test every format or validation branch and **does not simulate 500/503 failures**. Earlier isolated failure checks are recorded separately in [VALIDATION_REPORT.md](VALIDATION_REPORT.md); they are not part of this external client. Successful API tests establish the tested contract and recovery behavior, not model accuracy or exhaustive coverage.

## Standalone classification

To classify local images without running FastAPI:

```powershell
.\.venv\Scripts\python.exe -m scripts.classify_images examples/dog.jpg examples/vehicle.jpg
.\.venv\Scripts\python.exe -m scripts.classify_images examples/phone.jpg examples/cat.jpg
```

One classifier instance is reused for all paths in a run. The runner prints each filename and its actual `predictions` as readable JSON. It reports errors to stderr, continues to other paths, and exits nonzero if any path fails. The standalone result contains `predictions`; the HTTP API adds `model`.

## Included samples

| File | Description | Actual format | Dimensions | Bytes |
| --- | --- | --- | --- | ---: |
| `examples/dog.jpg` | Small tan dog on grass | JPEG | 600 × 400 | 42,622 |
| `examples/vehicle.jpg` | Silver sports car outdoors | JPEG | 1820 × 1025 | 398,513 |
| `examples/phone.jpg` | Phone sample | JPEG | 596 × 335 | 14,462 |
| `examples/cat.jpg` | Cat sample | JPEG | 847 × 1129 | 393,661 |

All four files decoded successfully and are below the per-file limit. Existing sample notes remain in [examples/SOURCES.md](examples/SOURCES.md).

## Limits and limitations

- Only one still JPEG or PNG image is accepted per prediction request. Batch uploads, image URLs, and base64/JSON image inputs are not implemented.
- The 5 MiB limit applies to uploaded file bytes **after multipart parsing**. The application reads at most the limit plus one byte from the parsed upload; this is not a complete HTTP request-body limit. It does not independently bound multipart overhead, decoded memory, or inference time. Existing Pillow decompression-bomb protections remain enabled.
- Integrity and decoding checks reject the tested malformed images, but are not exhaustive checks of every possible file defect.
- The model always chooses from its fixed 1,000 categories. It has no automatic `unknown` category or confidence rejection threshold. An unsupported subject can receive an unrelated label.
- Background, blur, lighting, viewpoint, resizing, and dataset bias can affect results. Scores do not guarantee correctness, and the project does not establish general accuracy, fairness, or safety-critical reliability.
- This is whole-image classification: it does not locate or count objects, return bounding boxes, segment regions, or identify an exact phone/vehicle model.
- The project has no custom training or deployment workflow. The documented server command is for local use.

## Project structure

Generated cache and virtual-environment contents are omitted:

```text
Image-Classification-API/
├── app/
│   ├── __init__.py
│   ├── main.py               # Lifespan, endpoints, upload checks, error handling
│   ├── model_service.py      # Reusable CPU classifier and preprocessing
│   └── schemas.py            # Pydantic response models
├── examples/
│   ├── dog.jpg
│   ├── vehicle.jpg
│   ├── phone.jpg
│   ├── cat.jpg
│   └── SOURCES.md            # Existing sample notes
├── scripts/
│   └── classify_images.py    # Standalone inference runner
├── test_api.py               # Requests-based HTTP test client
├── requirements.txt          # Exact installed dependency pins
├── PROJECT_SCOPE.md          # Historical scope notes
├── VALIDATION_REPORT.md      # Earlier isolated validation evidence
├── README.md                 # Current installation, usage, and reference
└── .gitignore                # Local environments, caches, bytecode, secrets
```

## Verification

Verified on 2026-11-03 using Windows x64, CPython 3.13.15, and the pinned dependencies. The HTTP client passed **10 checks with 0 failures and 0 unexecuted checks**, including successful prediction and health checks after invalid uploads. The dog and vehicle responses above matched real API responses exactly. A genuine PNG was accepted, while a truncated PNG and an additional uploaded file were rejected.

The four sample files decoded successfully. Python source syntax, installed dependency versions, and `pip check` passed. The Python upload example was executed successfully. Earlier isolated validation checks, including simulated 500/503 failures, are recorded in [VALIDATION_REPORT.md](VALIDATION_REPORT.md); those simulations are separate from the external test client.

Fresh installation and compatibility with other operating systems have not been reverified. The installation and standalone-runner commands were checked against the project files. These results establish the tested behavior, not model accuracy or exhaustive coverage.

## Demonstration Video

Video link: [Paste your demonstration video URL here]
