"""Check a running Image Classification API using real HTTP requests."""

import argparse
import math
from pathlib import Path
from urllib.parse import urlsplit

import requests


MODEL_ID = "google/vit-base-patch16-224"
MAX_UPLOAD_BYTES = 5_242_880
EXAMPLES = Path(__file__).resolve().parent / "examples"


class CheckFailed(Exception):
    """A request or response did not satisfy the API contract."""


class ServerUnavailable(CheckFailed):
    """The server cannot be reached, so further HTTP checks should stop."""


def expect(condition: bool, message: str, response: requests.Response) -> None:
    if not condition:
        excerpt = " ".join(response.text.split())[:240]
        raise CheckFailed(f"{message}; actual HTTP {response.status_code}; response: {excerpt!r}")


def request_json(session, base_url, timeout, method, path, expected_status, **kwargs):
    try:
        response = session.request(method, base_url + path, timeout=timeout, **kwargs)
    except requests.ConnectTimeout as exc:
        raise ServerUnavailable(f"expected HTTP {expected_status}; no response: server connection timed out ({exc})") from None
    except requests.Timeout as exc:
        raise CheckFailed(f"expected HTTP {expected_status}; no response: timeout after {timeout:g}s ({exc})") from None
    except requests.ConnectionError as exc:
        raise ServerUnavailable(f"expected HTTP {expected_status}; no response: server unreachable ({exc})") from None
    except requests.RequestException as exc:
        raise CheckFailed(f"expected HTTP {expected_status}; request failed ({exc})") from None
    expect(response.status_code == expected_status, f"expected HTTP {expected_status}", response)
    try:
        data = response.json()
    except ValueError:
        expect(False, f"expected HTTP {expected_status} with valid JSON", response)
    expect(isinstance(data, dict), f"expected HTTP {expected_status} with a JSON object", response)
    return response, data


def check_information(session, base_url, timeout):
    response, data = request_json(session, base_url, timeout, "GET", "/", 200)
    expect(data.get("name") == "Image Classification API", "expected HTTP 200 and API name", response)
    expect(data.get("model") == MODEL_ID, "expected HTTP 200 and model identifier", response)
    formats = data.get("supported_formats")
    expect(isinstance(formats, list) and "JPEG" in formats and "PNG" in formats,
           "expected HTTP 200 and JPEG/PNG support", response)


def check_health(session, base_url, timeout):
    response, data = request_json(session, base_url, timeout, "GET", "/health", 200)
    expect(data.get("status") == "ok" and data.get("model_loaded") is True,
           "expected HTTP 200 with ready classifier", response)


def check_prediction(session, base_url, timeout, path):
    try:
        with path.open("rb") as image:
            response, data = request_json(
                session, base_url, timeout, "POST", "/predict", 200,
                files={"file": (path.name, image, "image/jpeg")},
            )
    except OSError as exc:
        raise CheckFailed(f"expected HTTP 200; request not completed: sample file error ({exc})") from None
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    expect(content_type == "application/json", "expected HTTP 200 with application/json", response)
    expect(data.get("model") == MODEL_ID, "expected HTTP 200 and model identifier", response)
    predictions = data.get("predictions")
    expect(isinstance(predictions, list) and len(predictions) == 3,
           "expected HTTP 200 with exactly three predictions", response)
    scores = []
    for prediction in predictions:
        expect(isinstance(prediction, dict), "expected HTTP 200 with prediction objects", response)
        label = prediction.get("label")
        expect(isinstance(label, str) and bool(label.strip()),
               "expected HTTP 200 with nonempty labels", response)
        score = prediction.get("score")
        expect(isinstance(score, (int, float)) and not isinstance(score, bool)
               and 0 <= score <= 1 and math.isfinite(score),
               "expected HTTP 200 with finite numeric scores in [0, 1] (not booleans)", response)
        scores.append(score)
    expect(scores == sorted(scores, reverse=True), "expected HTTP 200 with descending scores", response)
    return f"top label={predictions[0]['label']!r}, score={scores[0]:.8f}"


def check_error(session, base_url, timeout, expected_status, files=None):
    response, data = request_json(
        session, base_url, timeout, "POST", "/predict", expected_status, files=files,
    )
    detail = data.get("detail")
    meaningful = (
        isinstance(detail, str) and bool(detail.strip())
        or isinstance(detail, dict) and bool(detail)
        or isinstance(detail, list) and bool(detail)
        and all(isinstance(item, dict) and isinstance(item.get("msg"), str)
                and bool(item["msg"].strip()) for item in detail)
    )
    expect(meaningful, f"expected HTTP {expected_status} with meaningful error detail", response)
    if expected_status == 413:
        text = str(detail).lower()
        expect(any(word in text for word in ("size", "limit", "mib", "large", "exceed")),
               "expected HTTP 413 explaining the size limit", response)


def positive_timeout(value):
    try:
        timeout = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return timeout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running API base URL")
    parser.add_argument("--timeout", type=positive_timeout, default=120.0,
                        help="Request connect/read timeout in seconds (default: 120)")
    args = parser.parse_args()
    try:
        parsed = urlsplit(args.base_url)
    except ValueError:
        parser.error("--base-url must be a valid HTTP(S) URL")
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error("--base-url must be an HTTP(S) URL without a query or fragment")
    base_url = args.base_url.rstrip("/")
    passed = failed = 0

    with requests.Session() as session:
        checks = [
            ("GET / information", lambda: check_information(session, base_url, args.timeout)),
            ("GET /health ready", lambda: check_health(session, base_url, args.timeout)),
            ("Predict dog JPEG", lambda: check_prediction(session, base_url, args.timeout, EXAMPLES / "dog.jpg")),
            ("Predict vehicle JPEG", lambda: check_prediction(session, base_url, args.timeout, EXAMPLES / "vehicle.jpg")),
            ("Missing upload returns 422", lambda: check_error(session, base_url, args.timeout, 422)),
            ("Text named invalid.jpg returns 400", lambda: check_error(
                session, base_url, args.timeout, 400,
                {"file": ("invalid.jpg", b"This is not an image", "image/jpeg")})),
            ("Empty upload returns 400", lambda: check_error(
                session, base_url, args.timeout, 400,
                {"file": ("empty.jpg", b"", "image/jpeg")})),
            ("Oversized upload returns 413", lambda: check_error(
                session, base_url, args.timeout, 413,
                {"file": ("oversized.jpg", b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")})),
            ("Valid prediction after invalid requests", lambda: check_prediction(
                session, base_url, args.timeout, EXAMPLES / "dog.jpg")),
            ("GET /health ready after invalid requests", lambda: check_health(session, base_url, args.timeout)),
        ]
        for name, check in checks:
            try:
                detail = check()
            except ServerUnavailable as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
                print(f"Stopped: server unreachable; {len(checks) - passed - failed} checks not executed.")
                break
            except CheckFailed as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
            else:
                passed += 1
                print(f"PASS {name}" + (f": {detail}" if detail else ""))

    print(f"Summary: {passed} passed, {failed} failed, {len(checks) - passed - failed} not executed.")
    return 0 if passed == len(checks) and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
