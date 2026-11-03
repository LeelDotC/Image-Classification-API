# Image Classification API — Scope

This document records the original design and project boundaries. The [README](README.md) describes the current implementation, installation, usage, and verified results.

## Purpose and assignment requirements

Provide an image-classification service using one pretrained Hugging Face model through FastAPI. The assignment requires:

- A selected Hugging Face model integrated into a FastAPI application.
- `GET /` for a brief system introduction.
- `GET /health` for operational status.
- Either `POST /predict` or `POST /generate` for appropriate input and structured JSON results.
- Basic input validation and handling of invalid data and inference failures.
- Complete source code, dependency instructions, README documentation, and a Python API test client using `requests`.
- A notebook only if Colab or Jupyter was used. This project uses local Python, so that condition does not apply.

## Selected model

**Model identifier:** `google/vit-base-patch16-224`

**Model card:** [Vision Transformer, base-sized](https://huggingface.co/google/vit-base-patch16-224)

The checkpoint classifies an entire image into one of 1,000 ImageNet-1k categories. It uses 16 × 16 image patches, was pretrained on ImageNet-21k, and was fine-tuned on ImageNet-1k. Its existing classification head and matching image processor support the project without custom training. The model card lists Apache-2.0.

The complete output category set is defined by [`id2label`](https://huggingface.co/google/vit-base-patch16-224/blob/main/config.json). Labels may contain comma-separated synonyms, which remain part of the same label.

## Project design choices

The assignment requires a prediction or generation endpoint; this project chooses **`POST /predict`**. The following upload and response details are project choices:

| Property | Contract |
| --- | --- |
| Request | `multipart/form-data` |
| Upload field | Exactly one file named `file` |
| Formats | Genuine JPEG or PNG; animated PNGs are rejected |
| Maximum uploaded file size | **5 MiB = 5,242,880 bytes**, inclusive, excluding multipart-envelope bytes |
| Success | HTTP 200 JSON with `model` and three `predictions` |
| Prediction entry | Complete model label and a finite score in [0, 1] |
| Ordering | Descending score |

Validation checks actual image content, integrity, and full decoding. Missing or extra files return 422; empty or corrupt images return 400; unsupported formats return 415; oversized files return 413. The implemented error behavior is documented in the README.

## Preprocessing and output interpretation

The classifier converts images to RGB and composites transparency onto white. The matching processor resizes to 224 × 224, rescales 8-bit pixel values to [0, 1], and normalizes each channel with mean 0.5 and standard deviation 0.5. See the [checkpoint processor configuration](https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json).

Softmax is applied across all 1,000 logits before selecting the highest three scores. The selected scores are not renormalized and need not sum to one. Scores are model estimates, not guarantees of correctness. Actual API responses are included in the README rather than duplicated here.

## Examples and boundaries

The dog and vehicle samples support the HTTP test client. The phone and cat samples provide additional inputs for standalone exploration. Their properties and existing source notes are documented in the README and [examples/SOURCES.md](examples/SOURCES.md).

This project performs whole-image classification, not detection, segmentation, caption generation, object counting, or exact product-model identification. The fixed label set cannot represent every subject, and there is no automatic unknown-category rejection. Background, image quality, viewpoint, resizing, and training-data coverage can affect results.

Batch uploads, URL/base64 image inputs, custom training, and deployment are outside the implemented scope. The upload-size check occurs after multipart parsing and is not a complete request-body or decoded-memory limit.
