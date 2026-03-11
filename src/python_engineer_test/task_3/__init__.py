"""Task 3: Car Path Tracking.

Detect and track moving cars from video, plot paths on map.
"""

from .car_tracking import detect_cars, track_cars, visualize_paths

__all__ = ["detect_cars", "track_cars", "visualize_paths"]
