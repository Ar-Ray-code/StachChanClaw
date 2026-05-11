#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from face_search import (
    DEFAULT_POSE_MODEL_URL,
    PAN_DEG_RANGE,
    TILT_DEG_RANGE,
    Detection,
    FaceRefiner,
    PoseRefiner,
    UpperBodyRefiner,
    V4L2MotorController,
    build_parser as build_face_search_parser,
    capture_after_settle,
    detect_snapshot,
    discover_last_sweep_index,
    ensure_asset,
    logical_to_degrees,
    open_camera,
    parse_float_list,
    parse_pair,
    read_fresh_frame,
    run_search_sweep,
    draw_debug,
)


OBJECT_CONFIG_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
OBJECT_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
OBJECT_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
INTERESTING_OBJECTS = {"person", "chair", "diningtable", "sofa", "tvmonitor", "pottedplant", "bottle"}
LOCAL_TZ = ZoneInfo("Asia/Tokyo")


@dataclass
class ObservationPaths:
    root_dir: Path
    run_dir: Path
    sweep_dir: Path
    frame_path: Path
    annotated_path: Path
    report_json_path: Path
    report_md_path: Path
    latest_frame_path: Path
    latest_annotated_path: Path
    latest_report_json_path: Path
    latest_report_md_path: Path
    history_jsonl_path: Path


class SceneObjectDetector:
    def __init__(
        self,
        asset_dir: Path,
        prototxt_path: Optional[Path],
        model_path: Optional[Path],
        confidence_threshold: float,
    ) -> None:
        prototxt = ensure_asset(asset_dir, prototxt_path, OBJECT_CONFIG_URL, "object detector config")
        model = ensure_asset(asset_dir, model_path, OBJECT_MODEL_URL, "object detector model")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame_h, frame_w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        self.net.setInput(blob)
        raw = self.net.forward()
        objects: list[Detection] = []
        for index in range(raw.shape[2]):
            score = float(raw[0, 0, index, 2])
            class_id = int(raw[0, 0, index, 1])
            if score < self.confidence_threshold or not (0 <= class_id < len(OBJECT_CLASSES)):
                continue
            label = OBJECT_CLASSES[class_id]
            if label not in INTERESTING_OBJECTS:
                continue
            left = float(np.clip(raw[0, 0, index, 3] * frame_w, 0.0, frame_w))
            top = float(np.clip(raw[0, 0, index, 4] * frame_h, 0.0, frame_h))
            right = float(np.clip(raw[0, 0, index, 5] * frame_w, 0.0, frame_w))
            bottom = float(np.clip(raw[0, 0, index, 6] * frame_h, 0.0, frame_h))
            if right <= left or bottom <= top:
                continue
            objects.append(Detection(score=score, x1=left, y1=top, x2=right, y2=bottom, kind=label))
        return objects


def build_parser():
    parser = build_face_search_parser()
    parser.description = "Run a single person-search observation and emit agent-friendly scene reports."
    parser.add_argument("--output-dir", type=Path, default=Path("captures/presence"), help="Root directory for latest artifacts and run history.")
    parser.add_argument("--timestamp", default=None, help="Override JST timestamp token for output paths.")
    parser.add_argument("--object-prototxt", type=Path, default=None, help="Path to an existing object detector prototxt.")
    parser.add_argument("--object-model", type=Path, default=None, help="Path to an existing object detector model.")
    parser.add_argument("--object-thresh", type=float, default=0.45, help="Confidence threshold for scene object detection.")
    parser.add_argument("--delete-cache", action="store_true", help="Do not save sweep screenshots; only write detection results to JSON.")
    return parser


def make_output_paths(root_dir: Path, timestamp: str) -> ObservationPaths:
    run_dir = root_dir / "runs" / timestamp
    sweep_dir = run_dir / "sweeps"
    run_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return ObservationPaths(
        root_dir=root_dir,
        run_dir=run_dir,
        sweep_dir=sweep_dir,
        frame_path=run_dir / "scene.jpg",
        annotated_path=run_dir / "scene_annotated.jpg",
        report_json_path=run_dir / "report.json",
        report_md_path=run_dir / "report.md",
        latest_frame_path=root_dir / "latest_frame.jpg",
        latest_annotated_path=root_dir / "latest_annotated.jpg",
        latest_report_json_path=root_dir / "latest_report.json",
        latest_report_md_path=root_dir / "latest_report.md",
        history_jsonl_path=root_dir / "history.jsonl"
    )


def serialize_detection(det: Detection) -> dict:
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


def expanded_contains(anchor: Detection, candidate: Detection, pad_ratio: float = 0.18) -> bool:
    pad_x = anchor.width * pad_ratio
    pad_y = anchor.height * pad_ratio
    return (
        anchor.x1 - pad_x <= candidate.cx <= anchor.x2 + pad_x
        and anchor.y1 - pad_y <= candidate.cy <= anchor.y2 + pad_y
    )


def associate_objects(human: Detection, objects: list[Detection]) -> list[dict]:
    associations: list[tuple[float, Detection]] = []
    for obj in objects:
        overlap_x1 = max(human.x1, obj.x1)
        overlap_y1 = max(human.y1, obj.y1)
        overlap_x2 = min(human.x2, obj.x2)
        overlap_y2 = min(human.y2, obj.y2)
        overlap_w = max(0.0, overlap_x2 - overlap_x1)
        overlap_h = max(0.0, overlap_y2 - overlap_y1)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / max(min(human.area, obj.area), 1.0)
        if overlap_ratio >= 0.05 or expanded_contains(human, obj):
            associations.append((overlap_ratio, obj))
    associations.sort(key=lambda item: (item[0], item[1].score), reverse=True)
    return [
        {
            "label": obj.kind,
            "score": round(float(obj.score), 6),
            "overlap_ratio": round(float(overlap_ratio), 4),
            "bbox": serialize_detection(obj)["bbox"],
        }
        for overlap_ratio, obj in associations
    ]


def posture_hint(human: Detection, associated_objects: list[dict], face_count: int, pose_count: int) -> str:
    aspect = human.width / max(human.height, 1e-6)
    labels = {item["label"] for item in associated_objects}
    if aspect >= 1.15:
        return "reclined"
    if "chair" in labels:
        return "seated"
    if "sofa" in labels:
        return "resting"
    if face_count > 0 and pose_count > 0:
        return "upright"
    if aspect <= 0.82:
        return "upright"
    return "uncertain"


def infer_time_slot(now_local: datetime) -> str:
    hour = now_local.hour
    if 5 <= hour < 10:
        return "morning"
    if 10 <= hour < 17:
        return "daytime"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def choose_primary_human(humans: list[Detection], target_kind: Optional[str]) -> Optional[int]:
    if not humans:
        return None
    if target_kind == "human_box":
        return max(range(len(humans)), key=lambda index: humans[index].score)
    return max(range(len(humans)), key=lambda index: humans[index].area)


def copy_latest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_markdown_report(report: dict) -> str:
    lines = [
        "# Camera Presence Report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- time_slot: {report['time_slot']}",
        f"- camera: {report['camera']['device']} ({report['camera']['width']}x{report['camera']['height']})",
        f"- presence: {report['summary']['presence']}",
        f"- state_hints: {', '.join(report['summary']['state_hints']) if report['summary']['state_hints'] else 'none'}",
        f"- target: {report['summary']['target_kind'] or 'none'}",
        f"- snapshot: {report['artifacts']['snapshot_path']}",
        f"- annotated: {report['artifacts']['annotated_path']}",
        f"- sweep_dir: {report['artifacts']['sweep_dir']}",
        "",
        "## Search",
        "",
        f"- found_target: {report['search']['found_target']}",
        f"- best_pan_deg: {report['search']['best_pan_deg']}",
        f"- best_tilt_deg: {report['search']['best_tilt_deg']}",
        f"- objective: {report['search']['objective']}",
        "",
        "## People",
        "",
    ]
    if report["scene"]["people"]:
        for person in report["scene"]["people"]:
            labels = ", ".join(item["label"] for item in person["associated_objects"]) or "none"
            lines.extend(
                [
                    f"- {person['id']}: posture={person['posture_hint']} score={person['box']['score']} faces={person['support']['faces']} poses={person['support']['poses']} upperbodies={person['support']['upperbodies']} objects={labels}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Scene Objects", ""])
    if report["scene"]["objects"]:
        for obj in report["scene"]["objects"]:
            lines.append(f"- {obj['kind']}: score={obj['score']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def append_history(paths: ObservationPaths, report: dict) -> None:
    paths.history_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.history_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    now_local = datetime.now(LOCAL_TZ)
    timestamp = args.timestamp or now_local.strftime("%Y%m%dT%H%M%S")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = make_output_paths(output_root, timestamp)

    pan_limits = parse_pair(args.pan_limits)
    tilt_limits = parse_pair(args.tilt_limits)
    deadzone_x, deadzone_y = parse_pair(args.deadzone)
    search_pan_route = parse_float_list(args.search_pan_route)
    search_step_degrees = parse_float_list(args.search_step_degrees)

    pose_model_path = ensure_asset(args.models_dir, args.pose_model, args.pose_model_url or DEFAULT_POSE_MODEL_URL, "pose model")
    face_refiner = FaceRefiner()
    pose_refiner = PoseRefiner(
        model_path=pose_model_path,
        min_detection_confidence=args.pose_min_detection_confidence,
        min_presence_confidence=args.pose_min_presence_confidence,
        min_visibility=args.pose_min_visibility,
        min_keypoints=args.pose_min_keypoints,
    )
    upperbody_refiner = UpperBodyRefiner()
    object_detector = SceneObjectDetector(
        asset_dir=args.models_dir,
        prototxt_path=args.object_prototxt,
        model_path=args.object_model,
        confidence_threshold=args.object_thresh,
    )
    controller = V4L2MotorController(
        device=args.camera_device,
        pan_value=args.start_pan,
        tilt_value=args.start_tilt,
        pan_limits=pan_limits,
        tilt_limits=tilt_limits,
        dry_run=args.dry_run,
    )
    cap = open_camera(args.camera_device, args.camera_width, args.camera_height, args.camera_fps)

    try:
        controller.move(controller.pan, controller.tilt, force=True)
        read_fresh_frame(cap, flush_reads=max(4, args.track_flush_reads))

        if not args.no_save_search_captures:
            os.makedirs(paths.sweep_dir, exist_ok=True)

        sweep_index = discover_last_sweep_index(paths.sweep_dir)
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
            capture_dir=None if args.no_save_search_captures else paths.sweep_dir,
            sweep_index=sweep_index + 1,
            human_min_area_ratio=args.human_min_area_ratio,
            human_max_area_ratio=args.human_max_area_ratio,
            human_min_aspect_ratio=args.human_min_aspect_ratio,
            human_max_aspect_ratio=args.human_max_aspect_ratio,
            min_votes_without_face=args.min_votes_without_face,
            delete_cache=args.delete_cache,
        )

        if search_result is None:
            controller.move(args.start_pan, args.start_tilt, force=True)
            frame = capture_after_settle(cap, args.search_interval, args.search_flush_reads)
            best_pan = logical_to_degrees(controller.pan, PAN_DEG_RANGE)
            best_tilt = logical_to_degrees(controller.tilt, TILT_DEG_RANGE)
            objective = None
            target_kind = None
        else:
            controller.move(search_result.pose.pan, search_result.pose.tilt, force=True)
            frame = capture_after_settle(cap, args.search_interval, args.search_flush_reads)
            best_pan = search_result.pose.pan_deg
            best_tilt = search_result.pose.tilt_deg
            objective = round(float(search_result.objective), 6)
            target_kind = search_result.target.kind

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
        scene_objects = object_detector.detect(frame)

        cv2.imwrite(str(paths.frame_path), frame)
        annotated = draw_debug(
            frame=frame,
            humans=snapshot.humans,
            faces=snapshot.faces,
            poses=snapshot.poses,
            upperbodies=snapshot.upperbodies,
            target=snapshot.target,
            smoothed_target=None,
            controller=controller,
            mode="presence_observer",
            centered=False,
            deadzone_x=deadzone_x,
            deadzone_y=deadzone_y,
            top_margin_px=args.top_margin,
            top_deadzone_px=args.top_deadzone,
            extra_lines=[
                f"time_slot={infer_time_slot(now_local)}",
                f"objects={len(scene_objects)}",
                f"pan_deg={best_pan:.1f} tilt_deg={best_tilt:.1f}",
            ],
        )
        for obj in scene_objects:
            color = (120, 80, 255) if obj.kind == "person" else (80, 220, 120)
            x1 = int(round(obj.x1))
            y1 = int(round(obj.y1))
            x2 = int(round(obj.x2))
            y2 = int(round(obj.y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                annotated,
                f"{obj.kind} {obj.score:.2f}",
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(paths.annotated_path), annotated)

        primary_human_index = choose_primary_human(snapshot.humans, target_kind)
        people = []
        for index, human in enumerate(snapshot.humans, start=1):
            faces = [face for face in snapshot.faces if expanded_contains(human, face, 0.10)]
            poses = [pose for pose in snapshot.poses if expanded_contains(human, pose, 0.10)]
            upperbodies = [upper for upper in snapshot.upperbodies if expanded_contains(human, upper, 0.10)]
            associated_objects = associate_objects(human, [obj for obj in scene_objects if obj.kind != "person"])
            people.append(
                {
                    "id": f"person_{index}",
                    "is_primary": primary_human_index == index - 1,
                    "box": serialize_detection(human),
                    "support": {
                        "faces": len(faces),
                        "poses": len(poses),
                        "upperbodies": len(upperbodies),
                    },
                    "posture_hint": posture_hint(human, associated_objects, len(faces), len(poses)),
                    "associated_objects": associated_objects,
                }
            )

        object_entries = [serialize_detection(obj) for obj in scene_objects]
        slot = infer_time_slot(now_local)
        presence = "present" if people else ("possible_present" if any(obj["kind"] == "person" for obj in object_entries) else "away")
        state_hints: list[str] = []
        if people:
            primary = next((person for person in people if person["is_primary"]), people[0])
            if slot == "morning" and primary["posture_hint"] in {"reclined", "resting"}:
                state_hints.append("possible_sleeping")
            elif primary["posture_hint"] == "seated":
                state_hints.append("possibly_seated")
        elif slot == "daytime":
            state_hints.append("possibly_out")
        if not state_hints and presence == "present":
            state_hints.append("person_detected")

        report = {
            "generated_at": now_local.isoformat(),
            "time_slot": slot,
            "camera": {
                "device": args.camera_device,
                "width": int(args.camera_width),
                "height": int(args.camera_height),
                "fps": int(args.camera_fps),
            },
            "search": {
                "found_target": search_result is not None,
                "best_pan_deg": round(float(best_pan), 3),
                "best_tilt_deg": round(float(best_tilt), 3),
                "objective": objective,
            },
            "summary": {
                "presence": presence,
                "target_kind": target_kind,
                "state_hints": state_hints,
                "primary_person_id": None if primary_human_index is None else people[primary_human_index]["id"],
            },
            "scene": {
                "people": people,
                "objects": object_entries,
                "human_candidates_raw": [serialize_detection(det) for det in snapshot.humans_raw],
                "faces": [serialize_detection(det) for det in snapshot.faces],
                "poses": [serialize_detection(det) for det in snapshot.poses],
                "upperbodies": [serialize_detection(det) for det in snapshot.upperbodies],
            },
            "artifacts": {
                "run_dir": str(paths.run_dir),
                "snapshot_path": str(paths.frame_path),
                "annotated_path": str(paths.annotated_path),
                "sweep_dir": str(paths.sweep_dir),
            },
        }

        report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        report_md = build_markdown_report(report)
        paths.report_json_path.write_text(report_json, encoding="utf-8")
        paths.report_md_path.write_text(report_md + "\n", encoding="utf-8")
        paths.latest_report_json_path.write_text(report_json, encoding="utf-8")
        paths.latest_report_md_path.write_text(report_md + "\n", encoding="utf-8")
        copy_latest(paths.frame_path, paths.latest_frame_path)
        copy_latest(paths.annotated_path, paths.latest_annotated_path)
        append_history(paths, report)

        print(report_md)
        return 0
    finally:
        pose_refiner.close()
        cap.release()


if __name__ == "__main__":
    sys.exit(main())
