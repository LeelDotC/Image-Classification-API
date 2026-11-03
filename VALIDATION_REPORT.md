# Validation and error-handling verification

Verified on 2026-11-03 using the existing CPython 3.13.15 environment and installed dependencies. A temporary in-process ASGI harness exercised the real application and CPU classifier in an isolated process. No running server was started, stopped, or modified.

## Validation behavior

- Uploads undergo Pillow integrity verification before reopening and full decoding. A PNG without its end chunk and a PNG with an invalid IDAT checksum return 400 before inference.
- Pillow images and temporary preprocessing images are explicitly closed, including when processing fails. The classifier handles RGB conversion, white compositing, and checkpoint preprocessing.

## Expected and actual responses

| Case | Expected | Actual |
| --- | ---: | ---: |
| Missing file | 422 | 422 |
| Empty file | 400 | 400 |
| Text named image.jpg | 400 | 400 |
| GIF named image.jpg | 415 | 415 |
| Truncated JPEG | 400 | 400 |
| Truncated PNG without end chunk | 400 | 400 |
| PNG with corrupt IDAT checksum | 400 | 400 |
| Valid JPEG | 200 | 200 |
| Valid vehicle JPEG | 200 | 200 |
| Valid PNG | 200 | 200 |
| Grayscale PNG | 200 | 200 |
| Transparent PNG | 200 | 200 |
| Animated PNG | 400 | 400 |
| Duplicate file fields | 422 | 422 |
| Additional uploaded file | 422 | 422 |
| Exactly 5,242,880 bytes | 200 | 200 |
| 5,242,881 bytes | 413 | 413 |
| JPEG with misleading MIME and extension | 200 | 200 |
| Classifier unavailable | 503 | 503 |
| Unexpected inference exception | 500 | 500 |

All 33 request checks matched their expected status. This includes 13 successful real-model recovery requests, one after every tested upload rejection and simulated runtime failure.

## Real inference and simulations

Successful JPEG requests used examples/dog.jpg and examples/vehicle.jpg. PNG, grayscale, transparent, and exact-limit JPEG requests also used the real model. Each success returned the model name and three nonempty mapped labels with finite scores in [0, 1], descending order, and valid JSON. Validation rejections did not invoke the classifier.

Only unavailable-classifier, unexpected-inference-exception, and startup-initialization-failure checks were simulated. State and patched methods were restored in finally/context-manager cleanup. Clients received generic errors; technical exceptions were captured in server-side logs. Initialization failure prevented startup, and shutdown cleared the stored model reference.

## Other verification

- Upload reads requested exactly 5,242,881 bytes, never an unbounded read.
- Boundary fixtures were supported, decodable JPEG data padded to 5,242,880 and 5,242,881 bytes; multipart-envelope bytes were excluded.
- Upload streams and opened Pillow images closed after success and failure.
- Pixel-level checks verified grayscale conversion, fully/partially transparent RGBA and LA compositing onto white, unchanged caller data, and temporary RGB-image cleanup after a processor exception.
- The same classifier instance was reused; only one constructor call occurred during successful lifespan startup.
- GET / and GET /health returned 200 while initialized; unavailable health returned 503.
- /docs and /openapi.json loaded, and prediction-response schema constraints remained present.
- Python syntax compilation and pip check passed.
- These isolated checks are separate from the external HTTP client in test_api.py.

## Limitations

The file-byte check runs after multipart parsing; it is not a complete request-body, decoded-memory, or execution-time limit. Existing Pillow resource safeguards remain enabled. These targeted temporary checks are not a complete automated API-test suite or an exhaustive image-file security audit.
