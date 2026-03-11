# Task 1: Drone Path Prediction

Predict a drone's motion path from a video using optical flow, and visualize the predicted path on a map.

## Input

- Video file (e.g. `video1.mp4`).
- The current implementation estimates a 2D path from frame-to-frame motion only (no real GPS is extracted).

## Usage

The flags below match `PYTHONPATH=src python -m python_engineer_test.cli task1 --help`.

```text
pet task1 [-h] [--output OUTPUT] [--method {lucas_kanade,farneback}] [--srt SRT] [--lat LAT] [--lon LON] [--altitude ALTITUDE] video
```

```bash
# Help
PYTHONPATH=src python -m python_engineer_test.cli task1 --help

# Run via installed CLI
pet task1 /path/to/video1.mp4 --output output/ --method lucas_kanade

# Run with SRT telemetry file for real GPS coordinates
pet task1 /path/to/video1.mp4 --output output/ --srt /path/to/telemetry.srt

# Or run via module
PYTHONPATH=src python -m python_engineer_test.cli task1 /path/to/video1.mp4 \
  --output output/ \
  --method farneback
```

## Methods

- `lucas_kanade` (default): Sparse optical flow over tracked feature points.
- `farneback`: Dense optical flow over the full frame.

## SRT Telemetry (Optional)

When an SRT telemetry file is provided via `--srt`, the tool extracts real GPS coordinates from the first telemetry packet and uses them as the starting location. Without SRT, use `--lat` and `--lon` to specify the starting coordinates.

### GPS Coordinates from Video

The test video (`video1.mp4`) was captured at:
- **Start**: (48.2658, 25.9185)
- **End**: (48.2659, 25.9193)

Run with the actual coordinates:
```bash
pet task1 video.mp4 --lat 48.2658 --lon 25.9185 --altitude 53.4

# Run with SRT telemetry file
pet task1 video.mp4 --srt video.SRT
```

## Output

The following files are created under `--output`:

- `path_coordinates.csv`: Predicted cumulative coordinates (`x,y`) in arbitrary motion units.
- `drone_path_map.html`: Folium HTML map plotting the predicted path (uses provided `--lat`/`--lon` or coordinates from `--srt`).

## Dependencies

```bash
pip install -e ".[task1]"
```

## Resources

- Task 1 video (`video1.mp4`): place at `resources/task1/video1.mp4`.
