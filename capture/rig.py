"""The rig loop: camera frames -> plate trigger -> face guard -> ``POST /plates/image``.

Run it with ``python -m capture --meal Dinner`` (see ``main`` for options and
environment variables). One line is printed per plate.

Frames never touch the disk: each captured frame is JPEG-encoded in memory and
sent to the API, or dropped. A frame the face guard refuses (a face, or no
working detector) is discarded before encoding.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import httpx

from capture.face_guard import FACE_MODEL_ENV, FaceGuard
from capture.source import FrameSource, open_source
from capture.trigger import PlateTrigger
from core.types import ImageBGR

Uploader = Callable[[bytes], str]
"""Sends one JPEG; returns a one-line outcome for the log. Must not raise for HTTP errors."""

DEFAULT_API_URL = "http://127.0.0.1:8000"


def encode_jpeg(image: ImageBGR, quality: int = 90) -> bytes:
    """JPEG bytes of a BGR frame, in memory.

    Raises:
        ValueError: OpenCV could not encode the image.
    """
    if image.size == 0:
        raise ValueError("cannot encode an empty frame")
    try:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except cv2.error as exc:
        raise ValueError(f"could not encode frame as JPEG: {exc}") from exc
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return bytes(buffer.tobytes())


def make_uploader(client: httpx.Client, api_url: str, token: str, meal: str) -> Uploader:
    """An ``Uploader`` posting multipart ``image`` + ``meal`` to ``{api_url}/plates/image``.

    Timeouts are the client's (``main`` uses 30 s: the API runs the models in-request).

    Raises:
        ValueError: empty token or meal (the API would refuse every plate).
    """
    if not token.strip():
        raise ValueError("LEFTOVER_DEVICE_TOKEN is empty; the API refuses unauthenticated plates")
    if not meal.strip():
        raise ValueError("meal must not be empty")
    url = api_url.rstrip("/") + "/plates/image"
    headers = {"Authorization": f"Bearer {token.strip()}"}

    def upload(jpeg: bytes) -> str:
        try:
            response = client.post(
                url,
                headers=headers,
                files={"image": ("plate.jpg", jpeg, "image/jpeg")},
                data={"meal": meal.strip()},
            )
        except httpx.HTTPError as exc:
            return f"not sent ({type(exc).__name__}: {exc})"
        if response.status_code == 200:
            body = response.json()
            return (
                f"stored plate_id={body.get('plate_id')} regions={body.get('regions_stored')} "
                f"for_review={body.get('regions_for_review')}"
            )
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return f"rejected HTTP {response.status_code}: {detail}"

    return upload


@dataclass(slots=True)
class RigStats:
    """Counts for one run of the loop."""

    frames: int = 0
    plates: int = 0
    sent: int = 0
    dropped: int = 0


def run_rig(
    source: FrameSource,
    trigger: PlateTrigger,
    guard: FaceGuard,
    upload: Uploader | None,
    *,
    out: Callable[[str], None] = print,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    max_plates: int | None = None,
    max_misses: int = 500,
    idle_sleep_s: float = 0.01,
    sleep: Callable[[float], None] = time.sleep,
) -> RigStats:
    """Read frames until the source dries up or ``max_plates`` plates were handled.

    Args:
        upload: None for a dry run (plates are reported, nothing is sent).
        max_misses: stop after this many consecutive empty reads (camera gone,
            or the end of a recorded video).

    Returns:
        Counts of frames read and plates captured, sent and dropped.
    """
    stats = RigStats()
    misses = 0
    while max_plates is None or stats.plates < max_plates:
        frame = source.read()
        if frame is None:
            misses += 1
            if misses >= max_misses:
                out(f"camera gave no frame {misses} times in a row; stopping")
                break
            sleep(idle_sleep_s)
            continue
        misses = 0
        stats.frames += 1
        captured = trigger.update(frame, clock())
        if captured is None:
            continue
        stats.plates += 1
        head = (
            f"plate {stats.plates} {captured.captured_at:%H:%M:%S} "
            f"sharpness={captured.sharpness:.0f}"
        )
        check = guard.check(captured.image)
        if check.has_face:
            stats.dropped += 1
            out(f"{head} -> dropped, not stored ({check.reason})")
            continue
        if upload is None:
            out(f"{head} -> dry run, not sent")
            continue
        try:
            jpeg = encode_jpeg(captured.image)
        except ValueError as exc:
            stats.dropped += 1
            out(f"{head} -> dropped ({exc})")
            continue
        result = upload(jpeg)
        if result.startswith("stored"):
            stats.sent += 1
        out(f"{head} -> {result}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m capture",
        description=(
            "Watch the dish return and send one face-free frame per plate to POST /plates/image. "
            "Env: LEFTOVER_CAMERA_SOURCE, LEFTOVER_DEVICE_TOKEN, LEFTOVER_FACE_MODEL, "
            "LEFTOVER_API_URL. Frames are never written to disk."
        ),
    )
    parser.add_argument("--meal", required=True, help="meal on today's menu, e.g. Dinner")
    parser.add_argument(
        "--api", help=f"API base URL (default $LEFTOVER_API_URL or {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--source", help="camera index or stream URL (default $LEFTOVER_CAMERA_SOURCE or 0)"
    )
    parser.add_argument("--dry-run", action="store_true", help="report plates, send nothing")
    parser.add_argument("--max-plates", type=int, default=None, help="stop after this many plates")
    parser.add_argument(
        "--still-ms", type=int, default=300, help="still time before capture (untuned)"
    )
    parser.add_argument(
        "--min-sharpness", type=float, default=50.0, help="Laplacian variance needed (untuned)"
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=4.0,
        help="mean abs diff counted as motion (untuned)",
    )
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)
    token = env.get("LEFTOVER_DEVICE_TOKEN", "")
    if not args.dry_run and not token.strip():
        print("LEFTOVER_DEVICE_TOKEN is not set; set it (same value as the API) or use --dry-run")
        return 2
    guard = FaceGuard.from_env(env)
    if not guard.available:
        print(f"WARNING: every plate will be dropped: {guard.unavailable_reason}")
    else:
        print(f"face guard: YuNet loaded ({FACE_MODEL_ENV})")
    try:
        source = open_source(args.source or env.get("LEFTOVER_CAMERA_SOURCE", "0"))
    except (RuntimeError, ValueError) as exc:
        print(exc)
        return 2
    trigger = PlateTrigger(
        motion_threshold=args.motion_threshold,
        still_ms=args.still_ms,
        min_sharpness=args.min_sharpness,
    )
    api_url = args.api or env.get("LEFTOVER_API_URL", DEFAULT_API_URL)
    print("learning the empty slot: keep it clear for a moment")
    try:
        with httpx.Client(timeout=30.0) as client:
            upload = None if args.dry_run else make_uploader(client, api_url, token, args.meal)
            stats = run_rig(source, trigger, guard, upload, max_plates=args.max_plates)
    except KeyboardInterrupt:
        print("stopped")
        return 0
    finally:
        source.close()
    print(
        f"{stats.plates} plates: {stats.sent} stored, {stats.dropped} dropped, "
        f"{stats.frames} frames read"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
