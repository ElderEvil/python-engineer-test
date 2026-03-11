"""Drone Path Prediction using Optical Flow.

This module implements:
1. Optical Flow algorithm for motion tracking
2. Path prediction from motion vectors
3. Map visualization
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from python_engineer_test.shared.srt_parser import load_srt_file

logger = logging.getLogger(__name__)


def extract_gps_path_from_srt(srt_path: str | Path) -> list[tuple[float, float]]:
    """Extract full GPS path from SRT telemetry file.

    Args:
        srt_path: Path to SRT telemetry file.

    Returns:
        List of (latitude, longitude) tuples from all SRT packets.
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    parser = load_srt_file(str(srt_path))
    gps_path: list[tuple[float, float]] = []

    for packet in parser.packets:
        lat = packet.get("latitude")
        lon = packet.get("longitude")
        if lat is not None and lon is not None:
            gps_path.append((lat, lon))

    logger.info(f"Extracted {len(gps_path)} GPS coordinates from SRT")
    return gps_path


def _validate_method(method: str) -> str:
    method = method.strip().lower()
    if method != "lucas_kanade":
        raise ValueError(f"Invalid method. Expected 'lucas_kanade', got: {method!r}")
    return method


def extract_motion_vectors(
    video_path: str | Path,
    method: str = "lucas_kanade",
    max_corners: int = 100,
) -> list[np.ndarray]:
    """Extract motion vectors from video using Optical Flow.

    Args:
        video_path: Path to input video file
        method: Optical Flow method (only 'lucas_kanade' supported)
        max_corners: Maximum number of corners to track (for Lucas-Kanade)

    Returns:
        List of motion vectors per frame
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    method = _validate_method(method)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")
    motion_vectors = []

    # Parameters for Lucas-Kanade optical flow
    lk_win_size = (15, 15)
    lk_max_level = 2
    lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read first frame (invalid/corrupt video?): {video_path}")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        old_gray,
        maxCorners=max_corners,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == "lucas_kanade" and p0 is not None and len(p0) > 0:
            next_pts = np.zeros_like(p0)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                p0,
                next_pts,
                winSize=lk_win_size,
                maxLevel=lk_max_level,
                criteria=lk_criteria,
            )

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Calculate motion vectors
                motion = good_new - good_old
                if len(motion) > 0:
                    motion_vectors.append(motion)
                else:
                    motion_vectors.append(np.zeros((0, 2)))

                # Update points for next iteration
                p0 = good_new.reshape(-1, 1, 2)

        old_gray = frame_gray.copy()

        # Re-detect features if too few remain
        if method == "lucas_kanade" and (p0 is None or len(p0) < 10):
            p0 = cv2.goodFeaturesToTrack(
                old_gray,
                maxCorners=max_corners,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7,
            )

    cap.release()
    return motion_vectors


# Maximum motion threshold to filter outliers (pixels per frame)
# Larger motions likely indicate scene changes, re-detections, or tracking errors
MAX_MOTION_THRESHOLD = 5.0


def predict_path(
    video_path: str | Path,
    method: str = "lucas_kanade",
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Predict drone path from video using motion estimation.

    Args:
        video_path: Path to input video file
        method: Optical Flow method
        output_path: Optional path to save coordinates

    Returns:
        Array of predicted coordinates (N x 2)
    """
    method = _validate_method(method)
    motion_vectors = extract_motion_vectors(video_path, method=method)

    path = np.array([[0.0, 0.0]])
    current_pos = np.array([0.0, 0.0])

    for motion in motion_vectors:
        if method == "lucas_kanade":
            if len(motion) > 0:
                delta = np.median(motion, axis=0)
                if np.linalg.norm(delta) <= MAX_MOTION_THRESHOLD:
                    current_pos = current_pos + delta
            path = np.vstack([path, current_pos])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, path, delimiter=",", header="x,y")

    return path


# Default altitude for scale calculation when no SRT available (from video1.mp4 start)
DEFAULT_ALTITUDE_M = 53.4


def visualize_path(
    path: np.ndarray,
    output_path: str | Path | None = None,
    map_center: tuple[float, float] | None = None,
    srt_path: str | Path | None = None,
    video_path: str | Path | None = None,
    lat: float | None = None,
    lon: float | None = None,
    altitude: float | None = None,
    gps_path: list[tuple[float, float]] | None = None,
) -> str:
    """Visualize drone path on a map.

    Args:
        path: Array of coordinates (N x 2) - used when no gps_path provided
        output_path: Path to save visualization (HTML for folium)
        map_center: Optional (lat, lon) center for map
        srt_path: Optional path to SRT telemetry file for real GPS data
        video_path: Optional path to video file for calculating accurate scale
        gps_path: Optional list of real (lat, lon) coordinates - bypasses pixel conversion
        lat: Starting latitude
        lon: Starting longitude
        altitude: Flight altitude in meters for scale calculation

    Returns:
        Path to generated visualization
    """
    _output_path: Path
    try:
        import folium
    except ImportError:
        raise ImportError("folium required: pip install -e '.[task1]'")

    # Use CLI altitude, or default from video
    _altitude = altitude if altitude is not None else DEFAULT_ALTITUDE_M

    # Priority: SRT > CLI > defaults
    if srt_path and os.path.exists(str(srt_path)):
        try:
            parser = load_srt_file(str(srt_path))
            first_packet = parser.packets[0] if parser.packets else None
            if first_packet:
                srt_lat = first_packet.get("latitude")
                srt_lon = first_packet.get("longitude")
                if srt_lat is not None and srt_lon is not None:
                    map_center = (srt_lat, srt_lon)
                    logger.info(f"Using SRT telemetry: {map_center}")

                srt_alt = first_packet.get("rel_alt") or first_packet.get("abs_alt")
                if srt_alt is not None:
                    _altitude = srt_alt
        except Exception as e:
            logger.warning(f"Failed to parse SRT file, using default altitude: {e}")
    elif lat is not None and lon is not None:
        map_center = (lat, lon)
    else:
        if map_center is None:
            map_center = (48.2658, 25.9185)

    # Use local variable to satisfy type checker
    final_map_center = map_center if map_center is not None else (48.2658, 25.9185)

    # Create map
    m = folium.Map(location=final_map_center, zoom_start=15)

    if gps_path:
        lat_lon_path = gps_path
        if lat_lon_path:
            final_map_center = lat_lon_path[0]
            m.location = list(final_map_center)
    else:
        # Simple scale calculation: approximate degrees per pixel
        # 111320 meters ≈ 1 degree latitude
        scale = _altitude / 111320.0

        # Convert pixel coordinates to lat/lon
        # Camera is rotated 90° clockwise, so we rotate the path to align with GPS north
        lat_lon_path = [
            (
                final_map_center[0] - p[0] * scale,  # -pixel_x → latitude (north)
                final_map_center[1] + p[1] * scale,  # pixel_y → longitude (east)
            )
            for p in path
        ]

    # Add path
    folium.PolyLine(
        lat_lon_path,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Drone Path",
    ).add_to(m)

    # Add dots along path (sample every N points to avoid too many markers)
    sample_rate = max(1, len(lat_lon_path) // 100)  # ~100 dots max
    for i in range(0, len(lat_lon_path), sample_rate):
        folium.CircleMarker(
            lat_lon_path[i],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    # Add start/end markers
    if len(lat_lon_path) > 0:
        folium.Marker(
            lat_lon_path[0],
            popup="Start",
            icon=folium.Icon(color="green"),
        ).add_to(m)
        folium.Marker(
            lat_lon_path[-1],
            popup="End",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    # Save map
    if output_path is None:
        output_path = "drone_path_map.html"

    _output_path = Path(output_path)
    _output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(_output_path))

    return str(_output_path)


def main(
    video_path: str,
    output_dir: str = "output",
    method: str = "lucas_kanade",
    srt_path: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    altitude: float | None = None,
) -> int:
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    method = _validate_method(method)

    logger.info(f"Processing video: {video_path}")

    gps_path: list[tuple[float, float]] | None = None
    if srt_path:
        logger.info(f"Using SRT telemetry: {srt_path}")
        try:
            gps_path = extract_gps_path_from_srt(srt_path)
            logger.info(f"Using real GPS path with {len(gps_path)} coordinates")
        except Exception as e:
            logger.warning(f"Failed to extract GPS from SRT: {e}")

    # Predict path (for CSV output, even when using real GPS for map)
    path = predict_path(
        video_path,
        method=method,
        output_path=_output_dir / "path_coordinates.csv",
    )
    logger.info(f"Predicted path with {len(path)} points")

    # Visualize with SRT data if available
    map_path = visualize_path(
        path,
        output_path=_output_dir / "drone_path_map.html",
        srt_path=srt_path,
        video_path=video_path,
        lat=lat,
        lon=lon,
        altitude=altitude,
        gps_path=gps_path,
    )
    logger.info(f"Map saved to: {map_path}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone Path Prediction")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument(
        "--method",
        default="lucas_kanade",
        choices=["lucas_kanade"],
        help="Optical flow method (only lucas_kanade supported)",
    )
    parser.add_argument(
        "--srt",
        default=None,
        help="Optional path to SRT telemetry file for real GPS and accurate scale",
    )
    args = parser.parse_args()

    main(
        args.video,
        args.output,
        method=args.method,
        srt_path=args.srt,
    )
