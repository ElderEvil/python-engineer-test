"""Car Detection and Tracking Module.

Implements:
1. Car detection using YOLO or background subtraction
2. Multi-object tracking
3. Path extraction and visualization
"""

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from python_engineer_test.shared.geo_utils import DroneCameraGeometry
from python_engineer_test.shared.srt_parser import load_srt_file

logger = logging.getLogger(__name__)


class CarDetector:
    """Car detector using YOLO or background subtraction."""

    def __init__(self, method: str = "background_subtraction", model_path: str | None = None):
        self.method = method
        self.model: Any = None
        self.bg_subtractor: Any = None

        if method == "yolo":
            self._init_yolo(model_path)
        elif method == "background_subtraction":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )

    def _init_yolo(self, model_path: str | None) -> None:
        """Initialize YOLO model."""
        try:
            import importlib

            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "ultralytics required for YOLO. Install with: pip install -e '.[task3]'"
            ) from e

        model_source = model_path or "yolov8n.pt"
        try:
            self.model = YOLO(model_source)
        except Exception as e:
            raise RuntimeError(
                "Failed to load YOLO model weights. If you're offline, use method "
                "'background_subtraction' or provide local weights (model_path)."
            ) from e

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect cars in frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.method == "yolo":
            return self._detect_yolo(frame)
        else:
            return self._detect_bg_subtraction(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect cars using YOLO."""
        results = self.model(frame, verbose=False)
        boxes = []

        for result in results:
            for box in result.boxes:
                # Filter for car, truck, bus classes (2, 3, 5, 7 in COCO)
                class_id = int(box.cls[0])
                if class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    boxes.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))

        return boxes

    def _detect_bg_subtraction(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect moving objects using background subtraction."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold and clean up
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size (cars should be reasonable size)
            if w > 50 and h > 30 and w < 400 and h < 300:
                boxes.append((x, y, w, h))

        return boxes


class CarTracker:
    """Multi-object tracker for cars using centroid tracking."""

    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        """Initialize tracker.

        Args:
            max_disappeared: Max frames before track is removed
            max_distance: Max distance for track association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks: dict[int, dict[str, Any]] = {}
        self.disappeared: dict[int, int] = {}

    def _get_centroid(self, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """Get centroid of bounding box."""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def update(
        self, detections: list[tuple[int, int, int, int]], frame_number: int = 0
    ) -> dict[int, tuple[int, int]]:
        """Update tracks with new detections.

        Args:
            detections: List of bounding boxes
            frame_number: Current frame number for path recording

        Returns:
            Dictionary of track_id -> centroid
        """
        # If no detections, mark all as disappeared
        if not detections:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return {tid: self._get_centroid(t["bbox"]) for tid, t in self.tracks.items()}

        # Compute centroids for new detections
        new_centroids = [self._get_centroid(d) for d in detections]

        # If no existing tracks, create new ones
        if not self.tracks:
            for i, det in enumerate(detections):
                self._create_track(det)
            return {tid: self._get_centroid(t["bbox"]) for tid, t in self.tracks.items()}

        # Match detections to tracks
        track_ids = list(self.tracks.keys())
        old_centroids = [self._get_centroid(t["bbox"]) for t in self.tracks.values()]

        # Compute distance matrix
        distances = np.zeros((len(old_centroids), len(new_centroids)))
        for i, oc in enumerate(old_centroids):
            for j, nc in enumerate(new_centroids):
                distances[i, j] = np.sqrt((oc[0] - nc[0]) ** 2 + (oc[1] - nc[1]) ** 2)

        # Greedy matching
        used_rows = set()
        used_cols = set()

        while len(used_rows) < len(old_centroids) and len(used_cols) < len(new_centroids):
            # Find minimum distance
            min_val = float("inf")
            min_row, min_col = -1, -1

            for i in range(len(old_centroids)):
                if i in used_rows:
                    continue
                for j in range(len(new_centroids)):
                    if j in used_cols:
                        continue
                    if distances[i, j] < min_val:
                        min_val = distances[i, j]
                        min_row, min_col = i, j

            if min_val > self.max_distance:
                break

            # Associate detection with track
            track_id = track_ids[min_row]
            self.tracks[track_id]["bbox"] = detections[min_col]
            self.tracks[track_id]["path"].append((*new_centroids[min_col], frame_number))
            self.disappeared[track_id] = 0

            used_rows.add(min_row)
            used_cols.add(min_col)

        # Handle unmatched tracks
        for i, track_id in enumerate(track_ids):
            if i not in used_rows:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in used_cols:
                self._create_track(det)

        return {tid: self._get_centroid(t["bbox"]) for tid, t in self.tracks.items()}

    def _create_track(self, bbox: tuple[int, int, int, int], frame_number: int = 0) -> None:
        centroid = self._get_centroid(bbox)
        self.tracks[self.next_id] = {
            "bbox": bbox,
            "path": [(*centroid, frame_number)],
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def get_paths(self) -> dict[int, list[tuple[int, int, int]]]:
        return {tid: t["path"] for tid, t in self.tracks.items()}


def detect_cars(video_path: str | Path, method: str = "background_subtraction") -> list[np.ndarray]:
    """Detect cars in video.

    Args:
        video_path: Path to video file
        method: Detection method

    Returns:
        List of detection arrays per frame
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = CarDetector(method=method)
    cap = cv2.VideoCapture(str(video_path))

    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        all_detections.append(detections)

    cap.release()
    return all_detections


def track_cars(
    video_path: str | Path,
    method: str = "background_subtraction",
    output_path: str | Path | None = None,
) -> dict[int, list[tuple[int, int, int]]]:
    """Track cars in video and extract paths.

    Args:
        video_path: Path to video file
        method: Detection method
        output_path: Optional path to save paths

    Returns:
        Dictionary of track_id -> list of (x, y, frame_number) tuples
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = CarDetector(method=method)
    tracker = CarTracker()

    cap = cv2.VideoCapture(str(video_path))
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracker.update(detections, frame_number)
        frame_number += 1

    cap.release()

    paths = tracker.get_paths()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for track_id, path in paths.items():
                for x, y, _frame in path:
                    f.write(f"{track_id},{x},{y}\n")

    return paths


def visualize_paths(
    paths: dict[int, list[tuple[int, int, int]]],
    output_path: str | Path | None = None,
    map_center: tuple[float, float] | None = None,
    srt_path: str | Path | None = None,
    video_path: str | Path | None = None,
) -> str:
    """Visualize car paths on a map.

    Args:
        paths: Dictionary of track_id -> list of (x, y, frame_number) tuples
        output_path: Path to save visualization
        map_center: Optional (lat, lon) center
        srt_path: Optional path to SRT file for GPS data
        video_path: Optional path to video for camera geometry

    Returns:
        Path to generated visualization
    """
    try:
        import folium
    except ImportError:
        raise ImportError("folium required: pip install -e '.[task3]'")

    # Default to Chernivtsi, Ukraine coordinates
    if map_center is None:
        map_center = (48.2917, 25.9358)

    scale_lon = 0.00001
    scale_lat = 0.00001
    srt_parser = None
    video_width = 1920
    video_height = 1080

    if srt_path and os.path.exists(str(srt_path)):
        try:
            srt_parser = load_srt_file(str(srt_path))
            if srt_parser.packets:
                first = srt_parser.packets[0]
                if first["latitude"] is not None and first["longitude"] is not None:
                    map_center = (first["latitude"], first["longitude"])

                if video_path and os.path.exists(str(video_path)) and first["rel_alt"] is not None:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()

                        if video_width > 0 and video_height > 0:
                            geometry = DroneCameraGeometry(
                                altitude_m=first["rel_alt"],
                                video_width_px=video_width,
                                video_height_px=video_height,
                                focal_length_mm=first.get("focal_len") or 24.0,
                            )
                            scale_lon, scale_lat = geometry.get_scale_degrees_per_pixel(
                                map_center[0]
                            )
        except Exception:
            pass

    m = folium.Map(location=map_center, zoom_start=15)

    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

    for i, (track_id, path) in enumerate(paths.items()):
        if len(path) < 2:
            continue

        color = colors[i % len(colors)]

        lat_lon_path = []
        for x, y, frame_number in path:
            if srt_parser is not None:
                timestamp = frame_number / 29.97  # Assuming 29.97 fps
                packet = srt_parser.interpolate_packet(timestamp)
                if packet and packet["latitude"] is not None and packet["longitude"] is not None:
                    frame_gps = (packet["latitude"], packet["longitude"])

                    center_x = video_width / 2
                    center_y = video_height / 2
                    dx = x - center_x
                    dy = -(y - center_y)  # Flip Y (image Y is down, lat is up)

                    lat = frame_gps[0] + dy * scale_lat
                    lon = frame_gps[1] + dx * scale_lon
                    lat_lon_path.append((lat, lon))
                    continue

            dx = x - video_width / 2
            dy = -(y - video_height / 2)
            lat = map_center[0] + dy * scale_lat
            lon = map_center[1] + dx * scale_lon
            lat_lon_path.append((lat, lon))

        if len(lat_lon_path) < 2:
            continue

        folium.PolyLine(
            lat_lon_path,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=f"Car {track_id}",
        ).add_to(m)

        folium.CircleMarker(
            lat_lon_path[0],
            radius=5,
            color=color,
            fill=True,
            popup=f"Car {track_id} Start",
        ).add_to(m)

    if output_path is None:
        output_path = "car_paths_map.html"

    _output_path = Path(output_path)
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(_output_path))

    return str(_output_path)


def main(
    video_path: str,
    output_dir: str = "output",
    method: str = "background_subtraction",
    srt_path: str | None = None,
) -> int:
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    _video_path = Path(video_path)

    if srt_path is None:
        candidate = _video_path.parent / "telemetry.srt"
        if candidate.exists():
            srt_path = str(candidate)

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Method: {method}")
    if srt_path:
        logger.info(f"Using SRT telemetry: {srt_path}")

    paths = track_cars(video_path, method=method, output_path=_output_dir / "car_paths.csv")
    logger.info(f"Tracked {len(paths)} cars")

    map_path = visualize_paths(
        paths,
        output_path=_output_dir / "car_paths_map.html",
        srt_path=srt_path,
        video_path=video_path,
    )
    logger.info(f"Map saved to: {map_path}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Car Path Tracking")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--method",
        "-m",
        default="background_subtraction",
        choices=["background_subtraction", "yolo"],
        help="Detection method",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="Optional path to SRT telemetry file for real GPS and accurate scale",
    )
    args = parser.parse_args()

    main(args.video, args.output, args.method, args.srt)
