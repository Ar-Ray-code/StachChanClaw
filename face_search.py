#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None


DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
PAN_DEG_RANGE = (0.0, 180.0)
TILT_DEG_RANGE = (70.0, 110.0)


@dataclass
class Detection:
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str
    votes: int = 1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class Target:
    kind: str
    x: float
    y: float
    box: Detection


@dataclass
class SearchPose:
    pan: int
    tilt: int
    pan_deg: float
    tilt_deg: float
    stage: str = "search"


@dataclass
class SearchResult:
    pose: SearchPose
    target: Target
    frame: np.ndarray
    humans: list[Detection]
    faces: list[Detection]
    poses: list[Detection]
    upperbodies: list[Detection]
    objective: float


@dataclass
class DetectionSnapshot:
    humans_raw: list[Detection]
    humans: list[Detection]
    faces: list[Detection]
    poses: list[Detection]
    upperbodies: list[Detection]
    target: Optional[Target]


@dataclass
class SweepImageReport:
    """Per-image detection result for sweep report JSON."""
    image_path: Optional[str]
    pose_index: int
    stage: str
    pan_deg: float
    tilt_deg: float
    step_deg: float
    humans_raw_count: int
    humans_count: int
    faces_count: int
    poses_count: int
    upperbodies_count: int
    target_kind: Optional[str]
    objective: Optional[float]
    humans: list[dict]
    faces: list[dict]
    poses: list[dict]
    upperbodies: list[dict]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return int(round(clamp(value, minimum, maximum)))


def degrees_to_logical(degrees: float, degree_range: tuple[float, float]) -> int:
    minimum, maximum = degree_range
    ratio = (degrees - minimum) / max(maximum - minimum, 1e-6)
    return clamp_int(ratio * 255.0, 0, 255)


def logical_to_degrees(logical_value: int, degree_range: tuple[float, float]) -> float:
    minimum, maximum = degree_range
    ratio = clamp(logical_value / 255.0, 0.0, 1.0)
    return minimum + ratio * (maximum - minimum)


def format_degree_token(value: float) -> str:
    return f"{value:05.1f}".replace("-", "m").replace(".", "p")


def parse_size(arg: str) -> tuple[int, int]:
    text = str(arg).lower().replace(" ", "")
    if "x" in text:
        height, width = text.split("x", 1)
        return int(float(height)), int(float(width))
    size = int(float(text))
    return size, size


def nms_detections(boxes: list[Detection], iou_thresh: float) -> list[Detection]:
    if not boxes:
        return boxes
    arr = np.array([[b.score, b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=np.float32)
    scores = arr[:, 0]
    x1 = arr[:, 1]
    y1 = arr[:, 2]
    x2 = arr[:, 3]
    y2 = arr[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        iou = inter / (areas[current] + areas[order[1:]] - inter + 1e-6)
        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]
    return [boxes[index] for index in keep]


def box_iou(a: Detection, b: Detection) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


def box_overlap_ratio(a: Detection, b: Detection) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    smaller = min(a.area, b.area)
    if smaller <= 1e-6:
        return 0.0
    return inter / smaller


def is_human_like_box(
    det: Detection,
    frame_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> bool:
    frame_h, frame_w = frame_shape
    area_ratio = det.area / max(frame_h * frame_w, 1.0)
    aspect_ratio = det.width / max(det.height, 1e-6)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False
    return True


def face_supports_human(face: Detection, human: Detection) -> bool:
    face_center_in_box = human.x1 <= face.cx <= human.x2 and human.y1 <= face.cy <= human.y1 + human.height * 0.72
    face_width_ratio = face.width / max(human.width, 1e-6)
    face_height_ratio = face.height / max(human.height, 1e-6)
    return face_center_in_box and 0.10 <= face_width_ratio <= 0.75 and 0.08 <= face_height_ratio <= 0.75


def pose_supports_human(pose: Detection, human: Detection) -> bool:
    pose_center_in_box = human.x1 <= pose.cx <= human.x2 and human.y1 <= pose.cy <= human.y2
    width_ratio = pose.width / max(human.width, 1e-6)
    height_ratio = pose.height / max(human.height, 1e-6)
    overlap_ratio = box_overlap_ratio(pose, human)
    return pose_center_in_box and overlap_ratio >= 0.25 and 0.15 <= width_ratio <= 1.15 and 0.15 <= height_ratio <= 1.15


def upperbody_supports_human(upperbody: Detection, human: Detection) -> bool:
    upper_center_in_box = human.x1 <= upperbody.cx <= human.x2 and human.y1 <= upperbody.cy <= human.y1 + human.height * 0.82
    width_ratio = upperbody.width / max(human.width, 1e-6)
    height_ratio = upperbody.height / max(human.height, 1e-6)
    return upper_center_in_box and 0.18 <= width_ratio <= 1.0 and 0.18 <= height_ratio <= 0.95


def validate_human_detections(
    humans: list[Detection],
    faces: list[Detection],
    poses: list[Detection],
    upperbodies: list[Detection],
    frame_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_votes_without_face: int,
) -> list[Detection]:
    filtered = [
        det
        for det in humans
        if is_human_like_box(det, frame_shape, min_area_ratio, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    ]
    if not filtered:
        return []

    if faces:
        face_backed = [det for det in filtered if any(face_supports_human(face, det) for face in faces)]
        if face_backed:
            return face_backed

    if poses:
        pose_backed = [det for det in filtered if any(pose_supports_human(pose, det) for pose in poses)]
        if pose_backed:
            return pose_backed

    if upperbodies:
        upperbody_backed = [det for det in filtered if any(upperbody_supports_human(upperbody, det) for upperbody in upperbodies)]
        if upperbody_backed:
            return upperbody_backed

    return [det for det in filtered if det.votes >= min_votes_without_face]


class FaceRefiner:
    def __init__(self, min_size: int = 24) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"failed to load Haar cascade: {cascade_path}")
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray, humans: list[Detection]) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        candidates: list[Detection] = []

        def detect_in_roi(x0: int, y0: int, x1: int, y1: int) -> None:
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                return
            faces = self.classifier.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_size, self.min_size),
            )
            for face_x, face_y, face_w, face_h in faces:
                candidates.append(
                    Detection(
                        score=1.0,
                        x1=float(x0 + face_x),
                        y1=float(y0 + face_y),
                        x2=float(x0 + face_x + face_w),
                        y2=float(y0 + face_y + face_h),
                        kind="face",
                    )
                )

        frame_h, frame_w = gray.shape[:2]
        if humans:
            for human in humans:
                pad_x = int(human.width * 0.08)
                pad_y = int(human.height * 0.08)
                x0 = clamp_int(human.x1 - pad_x, 0, frame_w)
                y0 = clamp_int(human.y1 - pad_y, 0, frame_h)
                x1 = clamp_int(human.x2 + pad_x, 0, frame_w)
                y1 = clamp_int(human.y1 + human.height * 0.65, 0, frame_h)
                detect_in_roi(x0, y0, x1, y1)
            if not candidates:
                detect_in_roi(0, 0, frame_w, frame_h)
        else:
            detect_in_roi(0, 0, frame_w, frame_h)

        return nms_detections(candidates, 0.3)


class UpperBodyRefiner:
    def __init__(self, min_size: int = 32) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_upperbody.xml"
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"failed to load Haar cascade: {cascade_path}")
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray, humans: list[Detection]) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        candidates: list[Detection] = []

        def detect_in_roi(x0: int, y0: int, x1: int, y1: int) -> None:
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                return
            bodies = self.classifier.detectMultiScale(
                roi,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(self.min_size, self.min_size),
            )
            for body_x, body_y, body_w, body_h in bodies:
                candidates.append(
                    Detection(
                        score=0.9,
                        x1=float(x0 + body_x),
                        y1=float(y0 + body_y),
                        x2=float(x0 + body_x + body_w),
                        y2=float(y0 + body_y + body_h),
                        kind="upperbody",
                    )
                )

        frame_h, frame_w = gray.shape[:2]
        if humans:
            for human in humans:
                x0 = clamp_int(human.x1, 0, frame_w)
                y0 = clamp_int(human.y1, 0, frame_h)
                x1 = clamp_int(human.x2, 0, frame_w)
                y1 = clamp_int(human.y1 + human.height * 0.9, 0, frame_h)
                detect_in_roi(x0, y0, x1, y1)
        if not candidates:
            detect_in_roi(0, 0, frame_w, frame_h)
        return nms_detections(candidates, 0.3)


class PoseRefiner:
    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_visibility: float = 0.45,
        min_keypoints: int = 8,
    ) -> None:
        if mp is None or mp_python is None or mp_vision is None:
            raise RuntimeError("mediapipe is required for pose verification. Run through `uv run` to use the project env.")
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_presence_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.min_visibility = min_visibility
        self.min_keypoints = min_keypoints

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        frame_h, frame_w = frame_bgr.shape[:2]
        candidates: list[Detection] = []
        for pose_landmarks in result.pose_landmarks:
            visible = [
                landmark
                for landmark in pose_landmarks
                if getattr(landmark, "visibility", 0.0) >= self.min_visibility
                and 0.0 <= landmark.x <= 1.0
                and 0.0 <= landmark.y <= 1.0
            ]
            if len(visible) < self.min_keypoints:
                continue
            xs = [landmark.x * frame_w for landmark in visible]
            ys = [landmark.y * frame_h for landmark in visible]
            pad_x = (max(xs) - min(xs)) * 0.18 + 8.0
            pad_y = (max(ys) - min(ys)) * 0.18 + 8.0
            candidates.append(
                Detection(
                    score=float(sum(landmark.visibility for landmark in visible) / len(visible)),
                    x1=clamp(min(xs) - pad_x, 0.0, float(frame_w)),
                    y1=clamp(min(ys) - pad_y, 0.0, float(frame_h)),
                    x2=clamp(max(xs) + pad_x, 0.0, float(frame_w)),
                    y2=clamp(max(ys) + pad_y, 0.0, float(frame_h)),
                    kind="pose",
                )
            )
        return nms_detections(candidates, 0.3)

    def close(self) -> None:
        self.landmarker.close()


class V4L2MotorController:
    def __init__(
        self,
        device: str,
        pan_value: int,
        tilt_value: int,
        pan_limits: tuple[int, int],
        tilt_limits: tuple[int, int],
        dry_run: bool,
    ) -> None:
        self.device = device
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.pan = clamp_int(pan_value, *pan_limits)
        self.tilt = clamp_int(tilt_value, *tilt_limits)
        self.dry_run = dry_run

    def move(self, pan: Optional[int] = None, tilt: Optional[int] = None, force: bool = False) -> bool:
        next_pan = self.pan if pan is None else clamp_int(pan, *self.pan_limits)
        next_tilt = self.tilt if tilt is None else clamp_int(tilt, *self.tilt_limits)
        if not force and next_pan == self.pan and next_tilt == self.tilt:
            return False

        ctrl_value = f"hue={next_pan},contrast={next_tilt}"
        if not self.dry_run:
            subprocess.run(
                ["v4l2-ctl", "-d", self.device, "--set-ctrl", ctrl_value],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.pan = next_pan
        self.tilt = next_tilt
        return True


def sample_route_degrees(
    route_degrees: list[float],
    step_degree: float,
) -> list[float]:
    route = [float(value) for value in route_degrees]
    if not route:
        raise ValueError("route_degrees must not be empty")
    if len(route) == 1:
        return [clamp(route[0], *PAN_DEG_RANGE)]
    step = abs(float(step_degree))
    if step <= 0.0:
        raise ValueError("step_degree must be > 0")

    sampled: list[float] = []
    for index, (start_deg, end_deg) in enumerate(zip(route, route[1:])):
        segment = [clamp(start_deg, *PAN_DEG_RANGE)]
        current = start_deg
        direction = 1.0 if end_deg >= start_deg else -1.0
        while abs(end_deg - current) > step:
            current += direction * step
            segment.append(clamp(current, *PAN_DEG_RANGE))
        segment.append(clamp(end_deg, *PAN_DEG_RANGE))
        if index > 0:
            segment = segment[1:]
        for pan_deg in segment:
            if sampled and abs(sampled[-1] - pan_deg) < 1e-6:
                continue
            sampled.append(pan_deg)
    return sampled


def build_refine_route(center_pan_deg: float, span_deg: float) -> list[float]:
    center = clamp(center_pan_deg, *PAN_DEG_RANGE)
    span = abs(float(span_deg))
    return [
        center,
        clamp(center - span, *PAN_DEG_RANGE),
        center,
        clamp(center + span, *PAN_DEG_RANGE),
        center,
    ]


def build_search_poses(
    pan_limits: tuple[int, int],
    tilt_limits: tuple[int, int],
    pan_route_degrees: list[float],
    tilt_degree: float,
    step_degree: float,
    stage: str,
) -> list[SearchPose]:
    sampled_pan_degrees = sample_route_degrees(pan_route_degrees, step_degree)
    tilt_deg = clamp(float(tilt_degree), *TILT_DEG_RANGE)
    poses: list[SearchPose] = []
    for pan_deg in sampled_pan_degrees:
        pan_deg = clamp(float(pan_deg), *PAN_DEG_RANGE)
        logical_pan = clamp_int(degrees_to_logical(pan_deg, PAN_DEG_RANGE), *pan_limits)
        logical_tilt = clamp_int(degrees_to_logical(tilt_deg, TILT_DEG_RANGE), *tilt_limits)
        resolved_pan_deg = logical_to_degrees(logical_pan, PAN_DEG_RANGE)
        resolved_tilt_deg = logical_to_degrees(logical_tilt, TILT_DEG_RANGE)
        if poses and poses[-1].pan == logical_pan and poses[-1].tilt == logical_tilt:
            continue
        poses.append(
            SearchPose(
                pan=logical_pan,
                tilt=logical_tilt,
                pan_deg=resolved_pan_deg,
                tilt_deg=resolved_tilt_deg,
                stage=stage,
            )
        )
    return poses


def ensure_asset(asset_dir: Path, asset_path: Optional[Path], asset_url: str, description: str) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    if asset_path is not None:
        if not asset_path.is_file():
            raise FileNotFoundError(f"{description} not found: {asset_path}")
        return asset_path

    parsed = urllib.parse.urlparse(asset_url)
    asset_name = Path(parsed.path).name
    target = asset_dir / asset_name
    if not target.exists():
        print(f"[INFO] downloading {description}: {asset_url}", flush=True)
        urllib.request.urlretrieve(asset_url, target)
    return target


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened() and device.startswith("/dev/video") and device[10:].isdigit():
        cap = cv2.VideoCapture(int(device[10:]), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera device: {device}")
    return cap


def read_fresh_frame(cap: cv2.VideoCapture, flush_reads: int = 2) -> np.ndarray:
    frame: Optional[np.ndarray] = None
    for _ in range(max(1, flush_reads)):
        ok, current = cap.read()
        if ok:
            frame = current
    if frame is None:
        raise RuntimeError("failed to read camera frame")
    return frame


def capture_after_settle(cap: cv2.VideoCapture, settle_sec: float, flush_reads: int) -> np.ndarray:
    if settle_sec > 0.0:
        time.sleep(settle_sec)
    return read_fresh_frame(cap, flush_reads=flush_reads)


def estimate_shift(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[tuple[float, float], float]:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hann = cv2.createHanningWindow((gray_a.shape[1], gray_a.shape[0]), cv2.CV_32F)
    return cv2.phaseCorrelate(gray_a * hann, gray_b * hann)


def choose_target(
    humans: list[Detection],
    faces: list[Detection],
    poses: list[Detection],
    top_margin_px: float,
    frame_shape: tuple[int, int],
) -> Optional[Target]:
    frame_h, frame_w = frame_shape
    center_x = frame_w * 0.5
    center_y = frame_h * 0.5

    def rank(det: Detection, prefer_human: bool) -> float:
        x_error = abs(det.cx - center_x) / max(frame_w, 1.0)
        y_error = abs(det.cy - center_y) / max(frame_h, 1.0)
        top_error = abs(det.y1 - top_margin_px) / max(frame_h, 1.0)
        area_bonus = det.area / max(frame_w * frame_h, 1.0)
        human_bonus = 0.08 if prefer_human else 0.0
        return det.score * 0.60 + area_bonus * 0.25 + human_bonus - x_error * 0.55 - y_error * 0.35 - top_error * 0.45

    if humans:
        human = max(humans, key=lambda item: rank(item, True))
        return Target("human_box", human.cx, human.cy, human)

    if poses:
        pose = max(poses, key=lambda item: rank(item, False))
        return Target("pose_box", pose.cx, pose.cy, pose)

    if faces:
        face = max(faces, key=lambda item: rank(item, False))
        return Target("face_box", face.cx, face.cy, face)

    return None


def detect_snapshot(
    frame: np.ndarray,
    face_refiner: FaceRefiner,
    pose_refiner: PoseRefiner,
    upperbody_refiner: UpperBodyRefiner,
    top_margin_px: int,
    human_min_area_ratio: float,
    human_max_area_ratio: float,
    human_min_aspect_ratio: float,
    human_max_aspect_ratio: float,
    min_votes_without_face: int,
) -> DetectionSnapshot:
    humans_raw: list[Detection] = []
    faces = face_refiner.detect(frame, humans_raw)
    poses = pose_refiner.detect(frame)
    upperbodies = upperbody_refiner.detect(frame, humans_raw)
    humans = validate_human_detections(
        humans_raw,
        faces,
        poses,
        upperbodies,
        frame.shape[:2],
        human_min_area_ratio,
        human_max_area_ratio,
        human_min_aspect_ratio,
        human_max_aspect_ratio,
        min_votes_without_face,
    )
    target = choose_target(humans, faces, poses, top_margin_px, frame.shape[:2])
    return DetectionSnapshot(
        humans_raw=humans_raw,
        humans=humans,
        faces=faces,
        poses=poses,
        upperbodies=upperbodies,
        target=target,
    )


def smooth_point(
    previous: Optional[tuple[float, float]],
    current: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    if previous is None:
        return current
    return (
        previous[0] * (1.0 - alpha) + current[0] * alpha,
        previous[1] * (1.0 - alpha) + current[1] * alpha,
    )


def compute_servo_delta(
    error_pixels: float,
    frame_extent: int,
    command_sign: int,
    gain: float,
    min_step: int,
    max_step: int,
) -> int:
    error_norm = error_pixels / max(frame_extent, 1)
    raw = command_sign * error_norm * gain
    delta = int(round(raw))
    if delta == 0 and abs(error_pixels) > 0:
        delta = command_sign * (1 if error_pixels > 0 else -1)
    magnitude = int(clamp(abs(delta), min_step, max_step))
    return 0 if error_pixels == 0 else magnitude * (1 if delta > 0 else -1)


def create_writer(record_path: Optional[Path], frame_shape: tuple[int, int], fps: int) -> Optional[cv2.VideoWriter]:
    if record_path is None:
        return None
    record_path.parent.mkdir(parents=True, exist_ok=True)
    frame_h, frame_w = frame_shape
    writer = cv2.VideoWriter(
        str(record_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {record_path}")
    return writer


def draw_labeled_box(
    image: np.ndarray,
    detection: Detection,
    color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    x1 = int(round(detection.x1))
    y1 = int(round(detection.y1))
    x2 = int(round(detection.x2))
    y2 = int(round(detection.y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{label_prefix} {detection.score:.2f} v{detection.votes}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_y = max(y1, text_h + baseline + 2)
    cv2.rectangle(image, (x1, text_y - text_h - baseline - 2), (x1 + text_w + 4, text_y + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug(
    frame: np.ndarray,
    humans: list[Detection],
    faces: list[Detection],
    poses: list[Detection],
    upperbodies: list[Detection],
    target: Optional[Target],
    smoothed_target: Optional[tuple[float, float]],
    controller: V4L2MotorController,
    mode: str,
    centered: bool,
    deadzone_x: int,
    deadzone_y: int,
    top_margin_px: int,
    top_deadzone_px: int,
    extra_lines: Optional[list[str]] = None,
) -> np.ndarray:
    vis = frame.copy()
    for human in humans:
        draw_labeled_box(vis, human, (255, 150, 0), "human")
    for face in faces:
        draw_labeled_box(vis, face, (0, 200, 0), "face")
    for pose in poses:
        draw_labeled_box(vis, pose, (0, 255, 255), "pose")
    for upperbody in upperbodies:
        draw_labeled_box(vis, upperbody, (180, 60, 255), "upper")
    frame_h, frame_w = vis.shape[:2]
    center_x = frame_w // 2
    center_y = frame_h // 2
    cv2.rectangle(
        vis,
        (center_x - deadzone_x, center_y - deadzone_y),
        (center_x + deadzone_x, center_y + deadzone_y),
        (200, 200, 200),
        1,
    )
    cv2.line(vis, (0, top_margin_px), (frame_w, top_margin_px), (0, 255, 255), 1)
    cv2.line(vis, (0, top_margin_px + top_deadzone_px), (frame_w, top_margin_px + top_deadzone_px), (0, 120, 120), 1)
    cv2.drawMarker(vis, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    if target is not None:
        cv2.rectangle(
            vis,
            (int(round(target.box.x1)), int(round(target.box.y1))),
            (int(round(target.box.x2)), int(round(target.box.y2))),
            (0, 0, 255),
            2,
        )
        cv2.drawMarker(vis, (int(target.x), int(target.y)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    if smoothed_target is not None:
        cv2.circle(vis, (int(smoothed_target[0]), int(smoothed_target[1])), 4, (0, 255, 255), -1)

    lines = [
        f"mode={mode} target={'none' if target is None else target.kind} centered={centered}",
        f"hue={controller.pan} contrast={controller.tilt}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    for index, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (8, 18 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search for a person/face box with WebcamChan and align it in frame.")
    parser.add_argument("--camera-device", default="/dev/video0", help="V4L2 camera device path.")
    parser.add_argument("--camera-width", type=int, default=320, help="Capture width.")
    parser.add_argument("--camera-height", type=int, default=240, help="Capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument("--pose-model", type=Path, default=None, help="Path to a MediaPipe pose landmarker `.task` model.")
    parser.add_argument("--pose-model-url", default=DEFAULT_POSE_MODEL_URL, help="Pose model URL to download when --pose-model is absent.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory for downloaded models.")
    parser.add_argument("--human-min-area-ratio", type=float, default=0.015, help="Reject human boxes smaller than this frame-area ratio.")
    parser.add_argument("--human-max-area-ratio", type=float, default=0.90, help="Reject human boxes larger than this frame-area ratio.")
    parser.add_argument("--human-min-aspect-ratio", type=float, default=0.28, help="Reject human boxes narrower than this width/height ratio.")
    parser.add_argument("--human-max-aspect-ratio", type=float, default=1.65, help="Reject human boxes wider than this width/height ratio.")
    parser.add_argument("--min-votes-without-face", type=int, default=2, help="Minimum detector votes required when no matching face, pose, or upper body is found.")
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.45, help="Minimum pose detector confidence.")
    parser.add_argument("--pose-min-presence-confidence", type=float, default=0.45, help="Minimum pose landmark presence confidence.")
    parser.add_argument("--pose-min-visibility", type=float, default=0.45, help="Minimum per-landmark visibility kept for the pose box.")
    parser.add_argument("--pose-min-keypoints", type=int, default=8, help="Minimum visible landmarks required to accept a pose.")
    parser.add_argument("--pan-limits", default="0,255", help="Logical limits for hue control.")
    parser.add_argument("--tilt-limits", default="0,255", help="Logical limits for contrast control.")
    parser.add_argument("--start-pan", type=int, default=128, help="Initial hue value.")
    parser.add_argument("--start-tilt", type=int, default=128, help="Initial contrast value.")
    parser.add_argument(
        "--search-pan-route",
        default="90,0,90,180,90",
        help="Pan search route in degrees for ID1. The route is walked in order.",
    )
    parser.add_argument(
        "--search-tilt-degree",
        type=float,
        default=110.0,
        help="Tilt degree for ID2 during search.",
    )
    parser.add_argument(
        "--search-step-degrees",
        default="30,15,5",
        help="Pan step sizes in degrees. Later stages refine around the best detected angle.",
    )
    parser.add_argument("--search-interval", type=float, default=0.60, help="Seconds to settle after each search move before capture.")
    parser.add_argument("--search-flush-reads", type=int, default=6, help="Number of frames to flush after each search move.")
    parser.add_argument("--rescan-interval", type=float, default=1.5, help="Seconds to wait at center before running the next sweep.")
    parser.add_argument("--lost-timeout", type=float, default=0.9, help="Start search again when target is missing for this many seconds.")
    parser.add_argument("--deadzone", default="24,18", help="Center deadzone in pixels: x,y.")
    parser.add_argument("--top-margin", type=int, default=4, help="Desired top margin in pixels for the detection box.")
    parser.add_argument("--top-deadzone", type=int, default=8, help="Allowed top-margin error in pixels.")
    parser.add_argument("--y-center-weight", type=float, default=0.5, help="Weight for box center Y alignment when correcting tilt.")
    parser.add_argument("--top-weight", type=float, default=0.5, help="Weight for keeping the box top near the upper edge.")
    parser.add_argument("--pan-gain", type=float, default=72.0, help="Tracking gain for hue.")
    parser.add_argument("--tilt-gain", type=float, default=60.0, help="Tracking gain for contrast.")
    parser.add_argument("--track-min-step", type=int, default=2, help="Minimum tracking step when correcting.")
    parser.add_argument("--track-max-step", type=int, default=18, help="Maximum tracking step when correcting.")
    parser.add_argument("--command-interval", type=float, default=0.18, help="Minimum seconds between tracking commands.")
    parser.add_argument("--track-settle-delay", type=float, default=0.25, help="Seconds to wait after a tracking move before capture.")
    parser.add_argument("--track-flush-reads", type=int, default=4, help="Number of frames to flush after a tracking move.")
    parser.add_argument("--smoothing", type=float, default=0.35, help="EMA alpha for target smoothing.")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV debug window.")
    parser.add_argument("--record", type=Path, default=None, help="Write a debug recording to this MP4 path.")
    parser.add_argument("--search-captures-dir", type=Path, default=Path("captures/search"), help="Directory where annotated search frames are saved.")
    parser.add_argument("--no-save-search-captures", action="store_true", help="Disable saving annotated search frames.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send hue/contrast commands.")
    parser.add_argument("--max-runtime", type=float, default=None, help="Exit after this many seconds.")
    return parser


def parse_pair(value: str) -> tuple[int, int]:
    left, right = str(value).split(",", 1)
    return int(left), int(right)


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def discover_last_sweep_index(capture_dir: Optional[Path]) -> int:
    if capture_dir is None or not capture_dir.exists():
        return 0
    last_index = 0
    for path in capture_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("sweep_"):
            continue
        try:
            last_index = max(last_index, int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return last_index


def _serialize_detection_for_report(det: Detection) -> dict:
    """Serialize Detection to dict for sweep report JSON."""
    return {
        "kind": det.kind,
        "score": round(float(det.score), 6),
        "votes": int(det.votes),
        "bbox": {
            "x1": round(float(det.x1), 2),
            "y1": round(float(det.y1), 2),
            "x2": round(float(det.x2), 2),
            "y2": round(float(det.y2), 2),
            "width": round(float(det.width), 2),
            "height": round(float(det.height), 2),
        },
        "center": {"x": round(float(det.cx), 2), "y": round(float(det.cy), 2)},
        "area": round(float(det.area), 2),
    }


def run_search_sweep(
    cap: cv2.VideoCapture,
    face_refiner: FaceRefiner,
    pose_refiner: PoseRefiner,
    upperbody_refiner: UpperBodyRefiner,
    controller: V4L2MotorController,
    pan_limits: tuple[int, int],
    tilt_limits: tuple[int, int],
    search_pan_route: list[float],
    search_tilt_degree: float,
    search_step_degrees: list[float],
    settle_sec: float,
    flush_reads: int,
    top_margin_px: int,
    top_deadzone_px: int,
    deadzone_x: int,
    deadzone_y: int,
    capture_dir: Optional[Path],
    sweep_index: int,
    human_min_area_ratio: float,
    human_max_area_ratio: float,
    human_min_aspect_ratio: float,
    human_max_aspect_ratio: float,
    min_votes_without_face: int,
    delete_cache: bool = False,
) -> Optional[SearchResult]:
    best_result: Optional[SearchResult] = None
    sweep_dir = None
    pose_counter = 0
    image_reports: list[SweepImageReport] = []
    saved_image_paths: list[Path] = []

    if capture_dir is not None:
        sweep_dir = capture_dir / f"sweep_{sweep_index:04d}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

    for stage_index, step_degree in enumerate(search_step_degrees):
        if stage_index == 0:
            poses = build_search_poses(
                pan_limits=pan_limits,
                tilt_limits=tilt_limits,
                pan_route_degrees=search_pan_route,
                tilt_degree=search_tilt_degree,
                step_degree=step_degree,
                stage="coarse",
            )
        else:
            if best_result is None:
                break
            poses = build_search_poses(
                pan_limits=pan_limits,
                tilt_limits=tilt_limits,
                pan_route_degrees=build_refine_route(best_result.pose.pan_deg, search_step_degrees[stage_index - 1]),
                tilt_degree=search_tilt_degree,
                step_degree=step_degree,
                stage=f"refine{stage_index}",
            )

        for pose in poses:
            pose_counter += 1
            controller.move(pose.pan, pose.tilt, force=True)
            frame = capture_after_settle(cap, settle_sec, flush_reads)
            snapshot = detect_snapshot(
                frame=frame,
                face_refiner=face_refiner,
                pose_refiner=pose_refiner,
                upperbody_refiner=upperbody_refiner,
                top_margin_px=top_margin_px,
                human_min_area_ratio=human_min_area_ratio,
                human_max_area_ratio=human_max_area_ratio,
                human_min_aspect_ratio=human_min_aspect_ratio,
                human_max_aspect_ratio=human_max_aspect_ratio,
                min_votes_without_face=min_votes_without_face,
            )
            objective = (
                compute_target_objective(snapshot.target, frame.shape[:2], top_margin_px)
                if snapshot.target is not None
                else float("-inf")
            )

            target_label = "none" if snapshot.target is None else snapshot.target.kind
            filename = (
                f"{pose_counter:02d}_{pose.stage}_pan{format_degree_token(pose.pan_deg)}_"
                f"tilt{format_degree_token(pose.tilt_deg)}_{target_label}.jpg"
            )
            image_path: Optional[str] = None

            if sweep_dir is not None:
                image_path = str(sweep_dir / filename)
                if not delete_cache:
                    annotated = draw_debug(
                        frame=frame,
                        humans=snapshot.humans,
                        faces=snapshot.faces,
                        poses=snapshot.poses,
                        upperbodies=snapshot.upperbodies,
                        target=snapshot.target,
                        smoothed_target=None,
                        controller=controller,
                        mode="search_eval",
                        centered=False,
                        deadzone_x=deadzone_x,
                        deadzone_y=deadzone_y,
                        top_margin_px=top_margin_px,
                        top_deadzone_px=top_deadzone_px,
                        extra_lines=[
                            f"sweep={sweep_index:04d} pose={pose_counter:02d} stage={pose.stage}",
                            f"pan={pose.pan_deg:.1f}deg tilt={pose.tilt_deg:.1f}deg step={step_degree:.1f}",
                            (
                                f"raw_humans={len(snapshot.humans_raw)} "
                                f"filtered={len(snapshot.humans)} "
                                f"faces={len(snapshot.faces)} "
                                f"poses={len(snapshot.poses)} upper={len(snapshot.upperbodies)}"
                            ),
                            "objective=none" if snapshot.target is None else f"objective={objective:.3f}",
                        ],
                    )
                    cv2.imwrite(image_path, annotated)
                    saved_image_paths.append(Path(image_path))

            report_entry = SweepImageReport(
                image_path=image_path,
                pose_index=pose_counter,
                stage=pose.stage,
                pan_deg=round(float(pose.pan_deg), 3),
                tilt_deg=round(float(pose.tilt_deg), 3),
                step_deg=round(float(step_degree), 3),
                humans_raw_count=len(snapshot.humans_raw),
                humans_count=len(snapshot.humans),
                faces_count=len(snapshot.faces),
                poses_count=len(snapshot.poses),
                upperbodies_count=len(snapshot.upperbodies),
                target_kind=target_label if snapshot.target is not None else None,
                objective=round(float(objective), 6) if snapshot.target is not None else None,
                humans=[_serialize_detection_for_report(det) for det in snapshot.humans],
                faces=[_serialize_detection_for_report(det) for det in snapshot.faces],
                poses=[_serialize_detection_for_report(det) for det in snapshot.poses],
                upperbodies=[_serialize_detection_for_report(det) for det in snapshot.upperbodies],
            )
            image_reports.append(report_entry)

            if snapshot.target is None:
                print(
                    f"[INFO] sweep stage={pose.stage} pan={pose.pan_deg:.1f}deg "
                    f"tilt={pose.tilt_deg:.1f}deg target=none",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] sweep stage={pose.stage} pan={pose.pan_deg:.1f}deg tilt={pose.tilt_deg:.1f}deg "
                    f"target={snapshot.target.kind} objective={objective:.3f}",
                    flush=True,
                )
                if best_result is None or objective > best_result.objective:
                    best_result = SearchResult(
                        pose=pose,
                        target=snapshot.target,
                        frame=frame,
                        humans=snapshot.humans,
                        faces=snapshot.faces,
                        poses=snapshot.poses,
                        upperbodies=snapshot.upperbodies,
                        objective=objective,
                    )

    # Write detection report JSON to sweep directory root
    if capture_dir is not None:
        detection_report = {
            "generated_at": datetime.now().isoformat(),
            "sweep_index": sweep_index,
            "total_images": len(image_reports),
            "images_saved": not delete_cache,
            "best_objective": round(float(best_result.objective), 6) if best_result else None,
            "best_target_kind": best_result.target.kind if best_result else None,
            "images": [
                {
                    "image_path": entry.image_path,
                    "pose_index": entry.pose_index,
                    "stage": entry.stage,
                    "pan_deg": entry.pan_deg,
                    "tilt_deg": entry.tilt_deg,
                    "step_deg": entry.step_deg,
                    "detection": {
                        "humans_raw_count": entry.humans_raw_count,
                        "humans_count": entry.humans_count,
                        "faces_count": entry.faces_count,
                        "poses_count": entry.poses_count,
                        "upperbodies_count": entry.upperbodies_count,
                        "target_kind": entry.target_kind,
                        "objective": entry.objective,
                    },
                    "detections": {
                        "humans": entry.humans,
                        "faces": entry.faces,
                        "poses": entry.poses,
                        "upperbodies": entry.upperbodies,
                    },
                }
                for entry in image_reports
            ],
        }
        report_path = capture_dir / "detection_report.json"
        report_path.write_text(json.dumps(detection_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return best_result


def compute_target_objective(
    target: Target,
    frame_shape: tuple[int, int],
    top_margin_px: int,
) -> float:
    frame_h, frame_w = frame_shape
    x_error = abs(target.box.cx - frame_w * 0.5) / max(frame_w, 1.0)
    y_error = abs(target.box.cy - frame_h * 0.5) / max(frame_h, 1.0)
    top_error = abs(target.box.y1 - top_margin_px) / max(frame_h, 1.0)
    area_bonus = target.box.area / max(frame_w * frame_h, 1.0)
    return target.box.score * 0.55 + area_bonus * 0.25 - x_error * 0.55 - y_error * 0.35 - top_error * 0.55


def main() -> int:
    args = build_parser().parse_args()
    pan_limits = parse_pair(args.pan_limits)
    tilt_limits = parse_pair(args.tilt_limits)
    deadzone_x, deadzone_y = parse_pair(args.deadzone)
    search_pan_route = parse_float_list(args.search_pan_route)
    search_step_degrees = parse_float_list(args.search_step_degrees)
    if len(search_pan_route) < 2:
        raise ValueError("--search-pan-route must contain at least two angles")
    if not search_step_degrees:
        raise ValueError("--search-step-degrees must contain at least one angle step")

    pose_model_path = ensure_asset(args.models_dir, args.pose_model, args.pose_model_url, "pose model")
    face_refiner = FaceRefiner()
    pose_refiner = PoseRefiner(
        model_path=pose_model_path,
        min_detection_confidence=args.pose_min_detection_confidence,
        min_presence_confidence=args.pose_min_presence_confidence,
        min_visibility=args.pose_min_visibility,
        min_keypoints=args.pose_min_keypoints,
    )
    upperbody_refiner = UpperBodyRefiner()
    controller = V4L2MotorController(
        device=args.camera_device,
        pan_value=args.start_pan,
        tilt_value=args.start_tilt,
        pan_limits=pan_limits,
        tilt_limits=tilt_limits,
        dry_run=args.dry_run,
    )
    cap = open_camera(args.camera_device, args.camera_width, args.camera_height, args.camera_fps)

    writer: Optional[cv2.VideoWriter] = None
    try:
        controller.move(controller.pan, controller.tilt, force=True)
        time.sleep(0.2)
        read_fresh_frame(cap, flush_reads=max(4, args.track_flush_reads))

        pan_command_sign = 1
        tilt_command_sign = 1

        last_target_time = 0.0
        last_command_time = 0.0
        last_log_time = 0.0
        next_rescan_time = time.monotonic()
        smoothed_target: Optional[tuple[float, float]] = None
        mode = "search"
        start_time = time.monotonic()
        sweep_index = discover_last_sweep_index(None if args.no_save_search_captures else args.search_captures_dir)
        frame = read_fresh_frame(cap, flush_reads=2)
        humans: list[Detection] = []
        faces: list[Detection] = []
        poses: list[Detection] = []
        upperbodies: list[Detection] = []
        target: Optional[Target] = None

        while True:
            now = time.monotonic()
            centered = False

            if mode != "align":
                if mode == "center_hold" and now < next_rescan_time:
                    frame = read_fresh_frame(cap, flush_reads=2)
                    snapshot = detect_snapshot(
                        frame=frame,
                        face_refiner=face_refiner,
                        pose_refiner=pose_refiner,
                        upperbody_refiner=upperbody_refiner,
                        top_margin_px=args.top_margin,
                        human_min_area_ratio=args.human_min_area_ratio,
                        human_max_area_ratio=args.human_max_area_ratio,
                        human_min_aspect_ratio=args.human_min_aspect_ratio,
                        human_max_aspect_ratio=args.human_max_aspect_ratio,
                        min_votes_without_face=args.min_votes_without_face,
                    )
                    humans = snapshot.humans
                    faces = snapshot.faces
                    poses = snapshot.poses
                    upperbodies = snapshot.upperbodies
                    target = snapshot.target
                    if target is not None:
                        mode = "align"
                        last_target_time = now
                else:
                    mode = "search"
                    sweep_index += 1
                    search_result = run_search_sweep(
                        cap=cap,
                        face_refiner=face_refiner,
                        pose_refiner=pose_refiner,
                        upperbody_refiner=upperbody_refiner,
                        controller=controller,
                        pan_limits=pan_limits,
                        tilt_limits=tilt_limits,
                        search_pan_route=search_pan_route,
                        search_tilt_degree=args.search_tilt_degree,
                        search_step_degrees=search_step_degrees,
                        settle_sec=args.search_interval,
                        flush_reads=args.search_flush_reads,
                        top_margin_px=args.top_margin,
                        top_deadzone_px=args.top_deadzone,
                        deadzone_x=deadzone_x,
                        deadzone_y=deadzone_y,
                        capture_dir=None if args.no_save_search_captures else args.search_captures_dir,
                        sweep_index=sweep_index,
                        human_min_area_ratio=args.human_min_area_ratio,
                        human_max_area_ratio=args.human_max_area_ratio,
                        human_min_aspect_ratio=args.human_min_aspect_ratio,
                        human_max_aspect_ratio=args.human_max_aspect_ratio,
                        min_votes_without_face=args.min_votes_without_face,
                    )
                    if search_result is None:
                        controller.move(args.start_pan, args.start_tilt, force=True)
                        frame = capture_after_settle(cap, args.search_interval, args.search_flush_reads)
                        snapshot = detect_snapshot(
                            frame=frame,
                            face_refiner=face_refiner,
                            pose_refiner=pose_refiner,
                            upperbody_refiner=upperbody_refiner,
                            top_margin_px=args.top_margin,
                            human_min_area_ratio=args.human_min_area_ratio,
                            human_max_area_ratio=args.human_max_area_ratio,
                            human_min_aspect_ratio=args.human_min_aspect_ratio,
                            human_max_aspect_ratio=args.human_max_aspect_ratio,
                            min_votes_without_face=args.min_votes_without_face,
                        )
                        humans = snapshot.humans
                        faces = snapshot.faces
                        poses = snapshot.poses
                        upperbodies = snapshot.upperbodies
                        target = snapshot.target
                        smoothed_target = None
                        mode = "center_hold"
                        next_rescan_time = time.monotonic() + args.rescan_interval
                    else:
                        controller.move(search_result.pose.pan, search_result.pose.tilt, force=True)
                        frame = capture_after_settle(cap, args.search_interval, args.search_flush_reads)
                        snapshot = detect_snapshot(
                            frame=frame,
                            face_refiner=face_refiner,
                            pose_refiner=pose_refiner,
                            upperbody_refiner=upperbody_refiner,
                            top_margin_px=args.top_margin,
                            human_min_area_ratio=args.human_min_area_ratio,
                            human_max_area_ratio=args.human_max_area_ratio,
                            human_min_aspect_ratio=args.human_min_aspect_ratio,
                            human_max_aspect_ratio=args.human_max_aspect_ratio,
                            min_votes_without_face=args.min_votes_without_face,
                        )
                        humans = snapshot.humans
                        faces = snapshot.faces
                        poses = snapshot.poses
                        upperbodies = snapshot.upperbodies
                        target = snapshot.target
                        smoothed_target = None
                        if target is None:
                            mode = "search"
                        else:
                            mode = "align"
                            last_target_time = time.monotonic()
            else:
                frame = read_fresh_frame(cap, flush_reads=2)
                snapshot = detect_snapshot(
                    frame=frame,
                    face_refiner=face_refiner,
                    pose_refiner=pose_refiner,
                    upperbody_refiner=upperbody_refiner,
                    top_margin_px=args.top_margin,
                    human_min_area_ratio=args.human_min_area_ratio,
                    human_max_area_ratio=args.human_max_area_ratio,
                    human_min_aspect_ratio=args.human_min_aspect_ratio,
                    human_max_aspect_ratio=args.human_max_aspect_ratio,
                    min_votes_without_face=args.min_votes_without_face,
                )
                humans = snapshot.humans
                faces = snapshot.faces
                poses = snapshot.poses
                upperbodies = snapshot.upperbodies
                target = snapshot.target

            if mode == "align" and target is not None:
                last_target_time = now
                smoothed_target = smooth_point(smoothed_target, (target.x, target.y), args.smoothing)
                target_x, target_y = smoothed_target
                frame_h, frame_w = frame.shape[:2]
                error_x = target_x - (frame_w * 0.5)
                error_y = target_y - (frame_h * 0.5)
                error_top = target.box.y1 - args.top_margin
                tilt_error = error_y * args.y_center_weight + error_top * args.top_weight
                centered = (
                    abs(error_x) <= deadzone_x
                    and abs(error_y) <= deadzone_y
                    and abs(error_top) <= args.top_deadzone
                )
                if not centered and now - last_command_time >= args.command_interval:
                    next_pan = controller.pan
                    next_tilt = controller.tilt
                    if abs(error_x) > deadzone_x:
                        next_pan += compute_servo_delta(
                            error_pixels=error_x,
                            frame_extent=frame_w,
                            command_sign=pan_command_sign,
                            gain=args.pan_gain,
                            min_step=args.track_min_step,
                            max_step=args.track_max_step,
                        )
                    if abs(error_y) > deadzone_y or abs(error_top) > args.top_deadzone:
                        next_tilt += compute_servo_delta(
                            error_pixels=tilt_error,
                            frame_extent=frame_h,
                            command_sign=tilt_command_sign,
                            gain=args.tilt_gain,
                            min_step=args.track_min_step,
                            max_step=args.track_max_step,
                        )
                    if controller.move(next_pan, next_tilt):
                        last_command_time = now
                        frame = capture_after_settle(cap, args.track_settle_delay, args.track_flush_reads)
                        snapshot = detect_snapshot(
                            frame=frame,
                            face_refiner=face_refiner,
                            pose_refiner=pose_refiner,
                            upperbody_refiner=upperbody_refiner,
                            top_margin_px=args.top_margin,
                            human_min_area_ratio=args.human_min_area_ratio,
                            human_max_area_ratio=args.human_max_area_ratio,
                            human_min_aspect_ratio=args.human_min_aspect_ratio,
                            human_max_aspect_ratio=args.human_max_aspect_ratio,
                            min_votes_without_face=args.min_votes_without_face,
                        )
                        humans = snapshot.humans
                        faces = snapshot.faces
                        poses = snapshot.poses
                        upperbodies = snapshot.upperbodies
                        target = snapshot.target
                        if target is not None:
                            smoothed_target = smooth_point(smoothed_target, (target.x, target.y), args.smoothing)
            elif mode == "align" and now - last_target_time >= args.lost_timeout:
                mode = "search"
                smoothed_target = None

            if now - last_log_time >= 1.0:
                target_label = "none" if target is None else target.kind
                print(
                    f"[INFO] mode={mode} target={target_label} humans={len(humans)} "
                    f"faces={len(faces)} poses={len(poses)} upper={len(upperbodies)} "
                    f"hue={controller.pan} contrast={controller.tilt}",
                    flush=True,
                )
                last_log_time = now

            if args.display or args.record is not None:
                vis = draw_debug(
                    frame=frame,
                    humans=humans,
                    faces=faces,
                    poses=poses,
                    upperbodies=upperbodies,
                    target=target,
                    smoothed_target=smoothed_target,
                    controller=controller,
                    mode=mode,
                    centered=centered,
                    deadzone_x=deadzone_x,
                    deadzone_y=deadzone_y,
                    top_margin_px=args.top_margin,
                    top_deadzone_px=args.top_deadzone,
                )
                if writer is None and args.record is not None:
                    writer = create_writer(args.record, vis.shape[:2], args.camera_fps)
                if writer is not None:
                    writer.write(vis)
                if args.display:
                    cv2.imshow("StachChanClaw Face Search", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

            if args.max_runtime is not None and (now - start_time) >= args.max_runtime:
                break

    except KeyboardInterrupt:
        print("[INFO] interrupted", flush=True)
    finally:
        pose_refiner.close()
        if writer is not None:
            writer.release()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
