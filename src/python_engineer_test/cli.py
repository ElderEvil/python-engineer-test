"""Main CLI entry point for Python Engineer Test tasks."""

import argparse
import logging
import os
import sys

from python_engineer_test.preflight import require_dir, require_file

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog="pet",
        description="Python Engineer Test - Task runner",
    )
    subparsers = parser.add_subparsers(dest="task", help="Task to run")

    # Task 1: Drone Path
    task1 = subparsers.add_parser("task1", help="Drone Path Prediction")
    task1.add_argument("video", help="Path to input video")
    task1.add_argument("--output", "-o", default="output", help="Output directory")
    task1.add_argument(
        "--method",
        "-m",
        default="lucas_kanade",
        choices=["lucas_kanade"],
        help="Optical Flow method (only lucas_kanade supported)",
    )
    task1.add_argument(
        "--srt",
        default=None,
        help="Optional path to SRT telemetry file for real GPS and accurate scale",
    )
    task1.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Starting latitude (default: 48.2658 from video1.mp4)",
    )
    task1.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Starting longitude (default: 25.9185 from video1.mp4)",
    )
    task1.add_argument(
        "--altitude",
        type=float,
        default=None,
        help="Flight altitude in meters for scale calculation (default: 53.4)",
    )

    # Task 2: Signal Bot
    task2 = subparsers.add_parser("task2", help="Signal Bot with ATAK Integration")
    task2.add_argument("phone", help="Signal phone number (with country code)")
    task2.add_argument("--atak-host", default="239.2.3.1", help="ATAK multicast address")
    task2.add_argument("--atak-port", type=int, default=6969, help="ATAK port")
    task2.add_argument(
        "--dry-run",
        metavar="MESSAGE",
        help="Parse one message and print CoT XML to stdout, then exit",
    )
    task2.add_argument(
        "--order",
        choices=["latlon", "lonlat"],
        default="lonlat",
        help="Coordinate order for input (default: lonlat, matches assignment PDF)",
    )

    # Task 3: Car Tracking
    task3 = subparsers.add_parser("task3", help="Car Path Tracking")
    task3.add_argument("video", help="Path to input video")
    task3.add_argument("--output", "-o", default="output", help="Output directory")
    task3.add_argument(
        "--method",
        "-m",
        default="background_subtraction",
        choices=["background_subtraction", "yolo"],
        help="Detection method",
    )
    task3.add_argument(
        "--srt",
        default=None,
        help="Optional path to SRT telemetry file for real GPS and accurate scale",
    )

    # Task 4: VIPE
    task4 = subparsers.add_parser("task4", help="VIPE Gaussian Splatting")
    task4.add_argument("dataset", help="Path to input dataset")
    task4.add_argument("--workspace", "-w", default="workspace", help="Workspace directory")
    task4.add_argument("--output", "-o", default="demo.mp4", help="Output video")
    task4.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without doing any work",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.task is None:
        parser.print_help()
        return 1

    debug = os.environ.get("PET_DEBUG") == "1"

    try:
        task_exit_code: int | None = None
        match args.task:
            case "task1":
                video_path = require_file(args.video)
                from python_engineer_test.task_1.drone_path import main as task1_main

                task_exit_code = task1_main(
                    str(video_path),
                    args.output,
                    method=args.method,
                    srt_path=args.srt,
                    lat=args.lat,
                    lon=args.lon,
                    altitude=args.altitude,
                )
            case "task2":
                from python_engineer_test.task_2.signal_bot import main as task2_main

                task_exit_code = task2_main(
                    args.phone,
                    args.atak_host,
                    args.atak_port,
                    dry_run=args.dry_run,
                    order=args.order,
                )
            case "task3":
                video_path = require_file(args.video)
                from python_engineer_test.task_3.car_tracking import main as task3_main

                task_exit_code = task3_main(str(video_path), args.output, args.method, args.srt)
            case "task4":
                from python_engineer_test.task_4.vipe_runner import main as task4_main

                dataset_dir = require_dir(args.dataset)
                task_exit_code = task4_main(
                    str(dataset_dir),
                    args.workspace,
                    args.output,
                    dry_run=args.dry_run,
                )
            case _:
                return 1

        if isinstance(task_exit_code, int):
            return task_exit_code
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        if debug:
            raise
        logger.error(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
