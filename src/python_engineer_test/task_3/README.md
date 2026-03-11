# Task 3: Car Path Tracking

Detect and track cars in video, plot paths on map.

## Usage

```bash
# Via CLI
pet task3 video.mp4 --output output/
pet task3 video.mp4 --method yolo
pet task3 video.mp4 --srt telemetry.srt

# Direct module
python -m python_engineer_test.task_3.car_tracking video.mp4
```

## Methods

| Method | Description |
|--------|-------------|
| `background_subtraction` | MOG2 background subtractor (default) |
| `yolo` | YOLOv8 object detection (requires ultralytics) |

## SRT Telemetry (Optional)

When an SRT telemetry file is provided via `--srt`, the tool uses per-frame GPS data for accurate geolocation:

```bash
pet task3 video.mp4 --srt telemetry.srt --output output/
```

## Output

- `car_paths.csv` - Tracked paths per car ID
- `car_paths_map.html` - Folium map with all car paths

## Tracking

Uses centroid tracking for multi-object association:
- Max disappeared frames: 30
- Max association distance: 100px

## Dependencies

```bash
pip install -e ".[task3]"
```

## Resources

- Video (video2): [Google Drive](https://drive.google.com/file/d/1lvqkeE9NOJBvUn8079WvrNQ9nOoVXLVT/view)
- SRT (srt2): [Google Drive](https://drive.google.com/file/d/1vu__-bpmzAhnEyxnxnk88W06-axVI7o7/view)
