from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.pipeline import PlatePipeline
from api.settings import Settings
from capture.face_guard import (
    DEFAULT_FACE_MODEL,
    FACE_MODEL_ENV,
    FaceGuard,
    contains_face,
    default_guard,
    yunet_detector,
)
from capture.rig import RigStats, encode_jpeg, main, make_uploader, run_rig
from capture.scale import (
    PLATE_DIAMETER_CM,
    PlateEllipse,
    detect_scale,
    fit_plate_ellipse,
    scale_from_ellipse,
    scale_from_plate,
    touches_border,
)
from capture.source import open_source, parse_source
from capture.trigger import PlateTrigger, TriggerState, sharpness
from core.types import DishMatch, FoodRegion, ImageBGR, Mask


def test_parse_source() -> None:
    assert parse_source(" 1 ") == 1
    assert parse_source("http://10.0.0.5:8080/video") == "http://10.0.0.5:8080/video"
    with pytest.raises(ValueError):
        parse_source("  ")


def test_sharp_image_scores_higher_than_its_blur() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(64, 64)).astype(np.uint8)
    blurred = (
        sharp[:-1, :-1].astype(np.float32) + sharp[1:, :-1] + sharp[:-1, 1:] + sharp[1:, 1:]
    ) / 4
    assert sharpness(sharp) > sharpness(blurred.astype(np.float32)) > 0.0
    assert sharpness(np.full((10, 10), 128, dtype=np.uint8)) == 0.0


def test_sharpness_rejects_colour_or_tiny_images() -> None:
    with pytest.raises(ValueError):
        sharpness(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        sharpness(np.zeros((2, 8), dtype=np.uint8))


FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def test_a_still_blank_scene_never_triggers() -> None:
    trigger = PlateTrigger()
    assert trigger.update(FRAME, datetime(2026, 11, 14, 18, 0)) is None
    assert feed(trigger, [FRAME] * 100) == []
    assert state(trigger) is TriggerState.EMPTY


def test_no_marker_means_no_scale() -> None:
    assert detect_scale(FRAME) is None


def test_aruco_marker_side_sets_the_scale() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 100)
    gray = np.full((300, 400), 255, dtype=np.uint8)
    gray[100:200, 150:250] = marker
    scale = detect_scale(np.repeat(gray[:, :, None], 3, axis=2), marker_size_cm=4.0)
    assert scale is not None
    # Detected corners sit on pixel centres: 99 px between them for a 100 px square.
    assert scale.cm_per_px == pytest.approx(4.0 / 100, rel=0.02)
    with pytest.raises(ValueError):
        detect_scale(gray)
    with pytest.raises(ValueError):
        detect_scale(FRAME, marker_size_cm=0.0)


def disc(
    shape: tuple[int, int] = (480, 640),
    center: tuple[int, int] = (320, 240),
    axes: tuple[int, int] = (160, 160),
    angle: float = 0.0,
) -> np.ndarray:
    """Filled ellipse mask; ``axes`` are semi-axes in px, as cv2.ellipse takes them."""
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(canvas, center, axes, angle, 0, 360, 1, -1)
    return canvas.astype(bool)


@pytest.mark.parametrize("radius", [40, 160, 230])
def test_plate_scale_from_a_full_disc(radius: int) -> None:
    scale = scale_from_plate(disc(axes=(radius, radius)))
    assert scale is not None
    # A disc drawn with semi-axis r spans 2r + 1 pixels; 1% covers rasterisation.
    assert scale.cm_per_px == pytest.approx(PLATE_DIAMETER_CM / (2 * radius + 1), rel=0.01)


def test_food_on_the_plate_does_not_change_the_scale() -> None:
    plate = disc()
    cv2.circle(plate.view(np.uint8), (320, 240), 70, 0, -1)  # food in the middle
    cv2.circle(plate.view(np.uint8), (460, 240), 45, 0, -1)  # food over the rim
    scale = scale_from_plate(plate)
    assert scale is not None
    assert scale.cm_per_px == pytest.approx(PLATE_DIAMETER_CM / 321, rel=0.01)


def test_tilted_camera_uses_the_major_axis() -> None:
    plate = disc(axes=(160, 140), angle=30.0)  # circle foreshortened along one axis
    ellipse = fit_plate_ellipse(plate)
    assert ellipse is not None
    assert ellipse.major_px == pytest.approx(321, rel=0.01)
    assert ellipse.minor_px == pytest.approx(281, rel=0.01)
    assert ellipse.center == pytest.approx((320, 240), abs=1.0)


def test_unusable_plates_give_no_scale() -> None:
    assert scale_from_plate(disc(center=(100, 240))) is None  # clipped by the left edge
    assert scale_from_plate(disc(axes=(8, 8))) is None  # too small to be a plate
    assert scale_from_plate(disc(axes=(200, 100))) is None  # too elongated
    square = np.zeros((480, 640), dtype=bool)
    square[100:380, 200:480] = True
    assert scale_from_plate(square) is None  # a tray, not a round plate
    assert scale_from_plate(np.zeros((480, 640), dtype=bool)) is None
    with pytest.raises(ValueError):
        scale_from_plate(np.zeros((4, 4, 3), dtype=bool))


def test_touches_border_and_bad_diameter() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    assert not touches_border(mask)
    mask[5, 8] = True
    assert touches_border(mask, margin_px=2) and not touches_border(mask, margin_px=1)
    with pytest.raises(ValueError):
        touches_border(np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError):
        scale_from_ellipse(PlateEllipse((0.0, 0.0), 10.0, 10.0, 0.0), plate_diameter_cm=0.0)


def test_blank_frame_has_no_face() -> None:
    seen: list[ImageBGR] = []

    def detector(image: ImageBGR) -> int:
        seen.append(image)
        return 0  # a fake YuNet that finds nothing, as YuNet should on a blank frame

    assert contains_face(FRAME, FaceGuard(detector)) is False
    assert len(seen) == 1 and seen[0] is FRAME


# --- camera source --------------------------------------------------------------


class FakeCapture:
    """Stands in for cv2.VideoCapture: replays (ok, frame) pairs."""

    def __init__(self, reads: list[tuple[bool, Any]], opened: bool = True) -> None:
        self.reads = list(reads)
        self.opened = opened
        self.released = 0
        self.target: int | str | None = None

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any]:
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self) -> None:
        self.released += 1


def test_open_source_wraps_the_capture_and_filters_bad_frames() -> None:
    fake = FakeCapture(
        [
            (True, FRAME),
            (False, None),
            (True, np.zeros((4, 4), np.uint8)),
            (True, FRAME.astype(np.float32)),
        ]
    )

    def opener(target: int | str) -> FakeCapture:
        fake.target = target
        return fake

    source = open_source(" 1 ", opener=opener)
    assert fake.target == 1
    first = source.read()
    assert first is not None and first.shape == (48, 64, 3)
    assert source.read() is None  # failed grab
    assert source.read() is None  # grey frame
    assert source.read() is None  # wrong dtype
    source.close()
    source.close()
    assert fake.released == 1
    assert source.read() is None


def test_open_source_refuses_a_camera_that_does_not_open() -> None:
    fake = FakeCapture([], opened=False)
    with pytest.raises(RuntimeError, match="cannot open camera source"):
        open_source("http://10.0.0.5:8080/video", opener=lambda _t: fake)
    assert fake.released == 1
    with pytest.raises(ValueError):
        open_source("")


def test_open_source_with_real_opencv_fails_cleanly_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        open_source(str(tmp_path / "no-such-video.avi"))


# --- plate trigger --------------------------------------------------------------

H, W = 240, 320
_, _XX = np.mgrid[0:H, 0:W]
EMPTY_SLOT = np.repeat((80 + _XX * 20 / W)[:, :, None].astype(np.uint8), 3, axis=2)
T0 = datetime(2026, 11, 14, 18, 0)


def plate_frame(
    center: tuple[int, int] = (160, 120), food: tuple[int, int, int] = (30, 90, 200)
) -> ImageBGR:
    """Overhead view of a white plate with two food items on the empty slot."""
    frame = EMPTY_SLOT.copy()
    cx, cy = center
    cv2.circle(frame, center, 70, (235, 235, 235), -1)
    cv2.circle(frame, (cx - 20, cy), 18, food, -1)
    cv2.ellipse(frame, (cx + 25, cy + 15), (20, 10), 30, 0, 360, (40, 160, 60), -1)
    return frame


def blurred(frame: ImageBGR, sigma: float) -> ImageBGR:
    return np.asarray(cv2.GaussianBlur(frame, (0, 0), sigma), dtype=np.uint8)


_NEXT_TIME: weakref.WeakKeyDictionary[PlateTrigger, datetime] = weakref.WeakKeyDictionary()


def feed(
    trigger: PlateTrigger, frames: list[ImageBGR], step_ms: int = 100
) -> list[tuple[int, Any]]:
    """Feed frames at a fixed rate, the clock carrying on across calls for the same trigger.

    Returns (index within ``frames``, capture) for every capture. The first
    frame ever fed to a trigger is stamped ``T0``.
    """
    now = _NEXT_TIME.get(trigger, T0)
    out = []
    for i, frame in enumerate(frames):
        captured = trigger.update(frame, now)
        if captured is not None:
            out.append((i, captured))
        now += timedelta(milliseconds=step_ms)
    _NEXT_TIME[trigger] = now
    return out


def state(trigger: PlateTrigger) -> TriggerState:
    """The trigger's state, read through a call so mypy does not narrow it between updates."""
    return trigger.state


def arrival(center: tuple[int, int] = (160, 120)) -> list[ImageBGR]:
    """A plate slid in from the left: three moving frames, then it rests at ``center``."""
    cx, cy = center
    return [plate_frame((cx - 60, cy)), plate_frame((cx - 30, cy)), plate_frame((cx - 10, cy))]


def test_one_sharp_frame_per_plate() -> None:
    trigger = PlateTrigger()
    resting = plate_frame()
    frames = [EMPTY_SLOT] * 10 + arrival() + [resting] * 40
    captures = feed(trigger, frames)
    assert len(captures) == 1
    index, captured = captures[0]
    assert index > 13  # not before it has been still for still_frames and still_ms
    assert np.array_equal(captured.image, resting)
    assert captured.sharpness >= trigger.min_sharpness
    assert captured.scale is None
    assert state(trigger) is TriggerState.CAPTURED


def test_capture_waits_for_still_ms_as_well_as_still_frames() -> None:
    trigger = PlateTrigger(still_ms=300, still_frames=5)
    frames = [EMPTY_SLOT] * 40 + arrival() + [plate_frame()] * 60  # 10 ms apart
    ((index, captured),) = feed(trigger, frames, step_ms=10)
    first_still = 43  # the first frame of the resting plate
    assert (index - first_still) * 10 >= 300  # 5 still frames alone would be 40 ms
    assert captured.captured_at >= T0 + timedelta(milliseconds=10 * first_still)


def test_rearms_only_after_the_slot_is_empty_again() -> None:
    trigger = PlateTrigger()
    first = [EMPTY_SLOT] * 10 + arrival() + [plate_frame()] * 15
    assert len(feed(trigger, first)) == 1
    # Swapped for another plate without the slot ever emptying: no second capture.
    other = plate_frame(food=(200, 40, 40))
    assert feed(trigger, [plate_frame((100, 120)), other, *[other] * 15]) == []
    assert state(trigger) is TriggerState.CAPTURED
    # Slot cleared, then a new plate: captured again.
    assert feed(trigger, [EMPTY_SLOT] * 10) == []
    assert state(trigger) is TriggerState.EMPTY
    assert len(feed(trigger, arrival() + [other] * 15)) == 1


def test_a_blurred_plate_is_not_captured_until_it_is_sharp() -> None:
    trigger = PlateTrigger()
    soft = blurred(plate_frame(), 3.0)
    assert (
        sharpness(np.asarray(cv2.cvtColor(soft, cv2.COLOR_BGR2GRAY), np.uint8))
        < trigger.min_sharpness
    )
    assert feed(trigger, [EMPTY_SLOT] * 10 + arrival() + [soft] * 30) == []
    assert state(trigger) is TriggerState.OCCUPIED
    sharp = plate_frame()
    ((_, captured),) = feed(trigger, [sharp] * 10)
    assert np.array_equal(captured.image, sharp)


def test_the_sharpest_frame_of_the_still_period_is_kept() -> None:
    trigger = PlateTrigger(still_ms=0, still_frames=3)
    sharp = plate_frame()
    slight = blurred(sharp, 1.0)
    frames = [EMPTY_SLOT] * 10 + [slight, sharp, slight, slight, slight]
    ((index, captured),) = feed(trigger, frames)
    assert index == 13
    assert np.array_equal(captured.image, sharp)
    assert captured.captured_at == T0 + timedelta(milliseconds=1100)


def test_an_arm_across_the_border_blocks_the_capture() -> None:
    trigger = PlateTrigger()
    with_arm = plate_frame()
    cv2.rectangle(with_arm, (0, 100), (110, 130), (150, 120, 110), -1)  # sleeve from the left
    assert feed(trigger, [EMPTY_SLOT] * 10 + arrival() + [with_arm] * 30) == []
    assert len(feed(trigger, [plate_frame()] * 10)) == 1


def test_a_plate_taken_away_before_capture_rearms_without_capturing() -> None:
    trigger = PlateTrigger()
    assert feed(trigger, [EMPTY_SLOT] * 10 + arrival() + [EMPTY_SLOT] * 5) == []
    assert state(trigger) is TriggerState.EMPTY


def test_slow_lighting_drift_is_absorbed_into_the_empty_slot() -> None:
    trigger = PlateTrigger()
    frames = [
        np.clip(EMPTY_SLOT.astype(np.int16) + i // 2, 0, 255).astype(np.uint8) for i in range(100)
    ]
    assert feed(trigger, frames) == []  # 50 grey levels brighter in the end
    assert state(trigger) is TriggerState.EMPTY


def test_a_new_resolution_relearns_the_empty_slot() -> None:
    trigger = PlateTrigger()
    feed(trigger, [EMPTY_SLOT] * 10)
    assert state(trigger) is TriggerState.EMPTY
    assert trigger.update(np.asarray(cv2.resize(EMPTY_SLOT, (200, 100)), np.uint8), T0) is None
    assert state(trigger) is TriggerState.LEARNING


def test_trigger_rejects_bad_frames_and_parameters() -> None:
    with pytest.raises(ValueError):
        PlateTrigger().update(np.zeros((48, 64), np.uint8), T0)
    with pytest.raises(ValueError):
        PlateTrigger().update(FRAME.astype(np.float32), T0)
    for bad in (
        {"motion_threshold": 0.0},
        {"still_frames": 0},
        {"empty_fraction": 0.1, "occupied_fraction": 0.05},
        {"background_alpha": 0.0},
        {"pixel_threshold": 300},
        {"work_width": 4},
    ):
        with pytest.raises(ValueError):
            PlateTrigger(**bad)


# --- face guard -----------------------------------------------------------------


def test_faces_found_by_the_detector_drop_the_frame() -> None:
    check = FaceGuard(lambda _image: 2).check(FRAME)
    assert check.has_face and "2 face" in check.reason


def test_a_missing_model_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "face_detection_yunet_2023mar.onnx"
    guard = FaceGuard.from_env({FACE_MODEL_ENV: str(missing)})
    assert not guard.available
    assert contains_face(FRAME, guard) is True
    check = guard.check(plate_frame())
    assert check.has_face and "fail closed" in check.reason and str(missing) in check.reason
    with pytest.raises(FileNotFoundError):
        yunet_detector(missing)


def test_unset_env_uses_the_default_path_and_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no models/ here
    guard = FaceGuard.from_env({})
    assert not guard.available and str(DEFAULT_FACE_MODEL) in guard.unavailable_reason


def test_the_default_guard_fails_closed_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FACE_MODEL_ENV, str(tmp_path / "missing.onnx"))
    default_guard.cache_clear()
    try:
        assert contains_face(FRAME) is True
    finally:
        default_guard.cache_clear()


def test_a_corrupt_model_fails_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "yunet.onnx"
    corrupt.write_bytes(b"not an onnx file")
    guard = FaceGuard.from_env({FACE_MODEL_ENV: str(corrupt)})
    assert not guard.available and "failed to load" in guard.unavailable_reason
    assert contains_face(FRAME, guard) is True


def test_detector_errors_and_bad_frames_fail_closed() -> None:
    def broken(_image: ImageBGR) -> int:
        raise RuntimeError("backend crashed")

    check = FaceGuard(broken).check(FRAME)
    assert check.has_face and "backend crashed" in check.reason
    ok = FaceGuard(lambda _image: 0)
    assert ok.check(np.zeros((48, 64), np.uint8)).has_face
    assert ok.check(np.zeros((0, 64, 3), np.uint8)).has_face
    assert ok.check(FRAME.astype(np.float32)).has_face
    assert not ok.check(FRAME).has_face


YUNET = Path("models/face_detection_yunet_2023mar.onnx")


@pytest.mark.skipif(
    not YUNET.is_file(), reason="YuNet weights not installed (see models/FACE_MODEL.md)"
)
def test_real_yunet_finds_no_face_in_a_blank_or_plate_frame() -> None:
    detect = yunet_detector(YUNET)
    assert detect(FRAME) == 0
    assert detect(np.zeros((1080, 1920, 3), np.uint8)) == 0  # downscaled path
    assert detect(plate_frame()) == 0


# --- rig loop -------------------------------------------------------------------


class ListSource:
    def __init__(self, frames: Sequence[ImageBGR | None]) -> None:
        self.frames = list(frames)
        self.closed = False

    def read(self) -> ImageBGR | None:
        return self.frames.pop(0) if self.frames else None

    def close(self) -> None:
        self.closed = True


def ticking_clock(step_ms: int = 100) -> Any:
    times: Iterator[int] = iter(range(0, 10**9, step_ms))
    return lambda: T0 + timedelta(milliseconds=next(times))


def two_plates() -> list[ImageBGR]:
    one = [EMPTY_SLOT] * 10 + arrival() + [plate_frame()] * 15
    two = [EMPTY_SLOT] * 10 + arrival() + [plate_frame(food=(200, 40, 40))] * 15
    return [*one, *two]


def run(
    frames: Sequence[ImageBGR | None], guard: FaceGuard, upload: Any, **kwargs: Any
) -> tuple[RigStats, list[str]]:
    lines: list[str] = []
    stats = run_rig(
        ListSource(frames),
        PlateTrigger(),
        guard,
        upload,
        out=lines.append,
        clock=ticking_clock(),
        max_misses=3,
        sleep=lambda _s: None,
        **kwargs,
    )
    return stats, lines


def test_rig_sends_one_jpeg_per_plate_and_writes_nothing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[bytes] = []

    def upload(jpeg: bytes) -> str:
        sent.append(jpeg)
        return "stored plate_id=abc regions=0 for_review=0"

    stats, lines = run(two_plates(), FaceGuard(lambda _i: 0), upload)
    assert (stats.plates, stats.sent, stats.dropped) == (2, 2, 0)
    assert len(sent) == 2
    decoded = cv2.imdecode(np.frombuffer(sent[0], np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (H, W, 3)
    plate_lines = [line for line in lines if line.startswith("plate ")]
    assert len(plate_lines) == 2 and "stored plate_id=abc" in plate_lines[0]
    assert lines[-1].startswith("camera gave no frame")
    assert list(tmp_path.iterdir()) == []


def test_rig_drops_frames_with_a_face_or_without_a_detector() -> None:
    calls: list[bytes] = []
    stats, lines = run(two_plates(), FaceGuard(lambda _i: 1), calls.append)
    assert (stats.plates, stats.sent, stats.dropped) == (2, 0, 2) and calls == []
    assert "dropped, not stored (1 face(s) detected)" in lines[0]
    stats, lines = run(two_plates(), FaceGuard(None, "YuNet model missing"), calls.append)
    assert stats.dropped == 2 and calls == [] and "fail closed: YuNet model missing" in lines[0]


def test_rig_dry_run_and_max_plates() -> None:
    stats, lines = run(two_plates(), FaceGuard(lambda _i: 0), None, max_plates=1)
    assert (stats.plates, stats.sent) == (1, 0)
    assert lines == [lines[0]] and lines[0].endswith("dry run, not sent")


def test_encode_jpeg_round_trips() -> None:
    jpeg = encode_jpeg(plate_frame())
    assert jpeg[:2] == b"\xff\xd8"
    with pytest.raises(ValueError):
        encode_jpeg(np.zeros((0, 0, 3), np.uint8))


def test_uploader_posts_multipart_with_the_device_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200, json={"plate_id": "p1", "regions_stored": 2, "regions_for_review": 1}
            )
        return httpx.Response(422, json={"detail": "no plate found in the photo"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        upload = make_uploader(client, "http://rig.test/", "s3cret", " Dinner ")
        assert upload(b"\xff\xd8jpeg") == "stored plate_id=p1 regions=2 for_review=1"
        assert upload(b"\xff\xd8jpeg") == "rejected HTTP 422: no plate found in the photo"
    request = requests[0]
    assert str(request.url) == "http://rig.test/plates/image"
    assert request.headers["Authorization"] == "Bearer s3cret"
    body = request.read()
    assert b'name="meal"' in body and b"Dinner" in body and b'name="image"' in body
    with pytest.raises(ValueError):
        make_uploader(httpx.Client(), "http://rig.test", " ", "Dinner")
    with pytest.raises(ValueError):
        make_uploader(httpx.Client(), "http://rig.test", "s3cret", "")


def test_uploader_reports_network_errors_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert make_uploader(client, "http://rig.test", "t", "Dinner")(b"x").startswith(
            "not sent (ConnectError"
        )


class CleanPlateSegmenter:
    def segment(self, image: ImageBGR) -> tuple[list[FoodRegion], Mask | None]:
        return [], np.ones(image.shape[:2], dtype=bool)


class NoMatcher:
    def match(self, image: ImageBGR, region: FoodRegion) -> DishMatch:
        return DishMatch(None, 0.0)


def test_rig_against_the_real_endpoint(tmp_path: Path) -> None:
    settings = Settings(
        hall="test-hall",
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        data_dir=tmp_path,
        device_token="device-secret",
        staff_token="staff-secret",
    )
    pipeline = PlatePipeline(
        segmenter=CleanPlateSegmenter(),
        measure=lambda _i, _p, _f: None,
        matcher_for=lambda _meal: NoMatcher(),
    )
    with TestClient(create_app(settings, pipeline=lambda: pipeline)) as client:
        stats, lines = run(
            two_plates(),
            FaceGuard(lambda _i: 0),
            make_uploader(client, "http://testserver", "device-secret", "Dinner"),
        )
        assert stats.sent == 2, lines
        wrong = make_uploader(client, "http://testserver", "staff-secret", "Dinner")
        assert wrong(encode_jpeg(plate_frame())).startswith("rejected HTTP 401")


def test_main_refuses_to_start_without_a_token_or_camera(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--meal", "Dinner"], env={}) == 2
    assert "LEFTOVER_DEVICE_TOKEN" in capsys.readouterr().out
    env = {
        FACE_MODEL_ENV: str(tmp_path / "missing.onnx"),
        "LEFTOVER_CAMERA_SOURCE": str(tmp_path / "none.avi"),
    }
    assert main(["--meal", "Dinner", "--dry-run"], env=env) == 2
    out = capsys.readouterr().out
    assert "every plate will be dropped" in out and "cannot open camera source" in out


def test_main_dry_run_on_a_recorded_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = tmp_path / "dish_return.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (W, H))
    assert writer.isOpened()
    for frame in two_plates():
        writer.write(frame)
    writer.release()
    env = {FACE_MODEL_ENV: str(tmp_path / "missing.onnx")}
    assert (
        main(["--meal", "Dinner", "--dry-run", "--still-ms", "0", "--source", str(clip)], env=env)
        == 0
    )
    out = capsys.readouterr().out
    # The clip is read through cv2.VideoCapture; without YuNet every plate is dropped.
    assert out.count("dropped, not stored (fail closed") == 2
    assert "2 plates: 0 stored, 2 dropped" in out
