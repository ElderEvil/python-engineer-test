"""VIPE Gaussian Splatting Runner.

This module handles:
1. Environment setup (local or RunPod)
2. Dataset preprocessing
3. VIPE pipeline execution
4. Result rendering
"""

import logging
import shlex
import subprocess
from pathlib import Path

from ..preflight import require_command

logger = logging.getLogger(__name__)


class VIPEManager:
    """Manager for VIPE Gaussian Splatting pipeline."""

    def __init__(
        self,
        workspace: str | Path = "workspace",
        vipe_repo: str = "https://github.com/nv-tlabs/vipe.git",
    ):
        """Initialize VIPE manager.

        Args:
            workspace: Directory for VIPE workspace
            vipe_repo: URL to VIPE repository
        """
        self.workspace = Path(workspace)
        self.vipe_repo = vipe_repo
        self.vipe_dir = self.workspace / "VIPE"

    def setup_environment(self) -> bool:
        """Set up VIPE environment.

        Returns:
            True if setup successful
        """
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Clone VIPE if not exists
        if not self.vipe_dir.exists():
            require_command("git")
            logger.info(f"Cloning VIPE to {self.vipe_dir}")
            result = subprocess.run(
                ["git", "clone", self.vipe_repo, str(self.vipe_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Failed to clone VIPE: {result.stderr}")
                return False

        # Create virtual environment
        venv_dir = self.workspace / "venv"
        if not venv_dir.exists():
            logger.info("Creating virtual environment...")
            subprocess.run(["python", "-m", "venv", str(venv_dir)], check=True)

        # Install dependencies
        requirements = self.vipe_dir / "requirements.txt"
        if requirements.exists():
            logger.info("Installing dependencies...")
            pip_path = venv_dir / "bin" / "pip"
            subprocess.run([str(pip_path), "install", "-r", str(requirements)], check=True)

        logger.info("Environment setup complete!")
        return True

    def prepare_dataset(
        self,
        dataset_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Prepare dataset for VIPE.

        Args:
            dataset_path: Path to input dataset
            output_dir: Output directory for prepared data

        Returns:
            Path to prepared dataset
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if output_dir is None:
            output_dir = self.workspace / "data"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # VIPE expects COLMAP format
        # This is a placeholder - actual preprocessing depends on dataset format
        logger.info(f"Dataset preparation would go here for {dataset_path}")
        logger.info(f"Output directory: {output_dir}")

        return output_dir

    def run_vipe(
        self,
        data_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Run VIPE Gaussian Splatting.

        Args:
            data_path: Path to prepared dataset
            output_path: Path for output model

        Returns:
            Path to output model
        """
        data_path = Path(data_path)
        if output_path is None:
            output_path = self.workspace / "output"

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run VIPE (placeholder - actual command depends on VIPE setup)
        logger.info(f"Running VIPE on {data_path}")
        logger.info(f"Output to {output_path}")

        # Example command structure
        cmd = [
            "python",
            str(self.vipe_dir / "train.py"),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
        ]

        logger.info(f"Would run: {' '.join(cmd)}")
        logger.info("Note: Actual VIPE execution requires proper GPU setup")

        return output_path

    def render_demo(
        self,
        model_path: str | Path,
        output_video: str | Path = "demo.mp4",
        trajectory: str = "orbit",
    ) -> Path:
        """Render demo video from Gaussian Splatting model.

        Args:
            model_path: Path to trained model
            output_video: Output video path
            trajectory: Camera trajectory type

        Returns:
            Path to output video
        """
        model_path = Path(model_path)
        output_video = Path(output_video)

        logger.info(f"Rendering demo from {model_path}")
        logger.info(f"Trajectory: {trajectory}")
        logger.info(f"Output: {output_video}")

        # Placeholder - actual rendering depends on VIPE structure
        return output_video


def _format_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _print_dry_run_plan(
    *,
    manager: VIPEManager,
    dataset_path: Path,
    output_video: str | Path,
) -> None:
    output_video = Path(output_video)

    logger.info("DRY RUN - no actions will be executed")
    logger.info("\nPlan:")
    logger.info(f"- Dataset: {dataset_path}")
    logger.info(f"- Workspace: {manager.workspace}")
    logger.info(f"- Output video: {output_video}")

    commands: list[list[str]] = []

    if not manager.vipe_dir.exists():
        require_command("git")
        commands.append(["git", "clone", manager.vipe_repo, str(manager.vipe_dir)])

    venv_dir = manager.workspace / "venv"
    if not venv_dir.exists():
        commands.append(["python", "-m", "venv", str(venv_dir)])

    requirements = manager.vipe_dir / "requirements.txt"
    pip_path = venv_dir / "bin" / "pip"
    commands.append([str(pip_path), "install", "-r", str(requirements)])

    prepared_data = manager.workspace / "data"
    model_out = manager.workspace / "output"
    commands.append(
        [
            "python",
            str(manager.vipe_dir / "train.py"),
            "--data",
            str(prepared_data),
            "--output",
            str(model_out),
        ]
    )

    logger.info("\nCommands (copy/paste):")
    logger.info(f"$ mkdir -p {shlex.quote(str(manager.workspace))}")
    for cmd in commands:
        logger.info(f"$ {_format_cmd(cmd)}")

    logger.info("\nNotes:")
    logger.info(f"- Dataset preparation is a placeholder; would write to: {prepared_data}")
    logger.info(f"- Training output directory would be: {model_out}")
    logger.info(f"- Rendering is a placeholder; would write video to: {output_video}")


def run_vipe_pipeline(
    dataset_path: str | Path,
    workspace: str | Path = "workspace",
    output_video: str = "demo.mp4",
    dry_run: bool = False,
) -> dict[str, Path]:
    """Run complete VIPE pipeline.

    Args:
        dataset_path: Path to input dataset
        workspace: Workspace directory
        output_video: Output demo video name

    Returns:
        Dictionary with paths to outputs
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    manager = VIPEManager(workspace=workspace)

    if dry_run:
        _print_dry_run_plan(manager=manager, dataset_path=dataset_path, output_video=output_video)
        return {
            "workspace": manager.workspace,
            "model": manager.workspace / "output",
            "video": Path(output_video),
        }

    # Setup
    if not manager.setup_environment():
        raise RuntimeError("Failed to setup environment")

    # Prepare dataset
    data_path = manager.prepare_dataset(dataset_path)

    # Run VIPE
    model_path = manager.run_vipe(data_path)

    # Render demo
    video_path = manager.render_demo(model_path, output_video)

    return {
        "workspace": manager.workspace,
        "model": model_path,
        "video": video_path,
    }


def main(
    dataset_path: str,
    workspace: str = "workspace",
    output: str = "demo.mp4",
    dry_run: bool = False,
) -> int:
    logger.info("=" * 50)
    logger.info("VIPE Gaussian Splatting Pipeline")
    logger.info("=" * 50)

    try:
        results = run_vipe_pipeline(
            dataset_path=dataset_path,
            workspace=workspace,
            output_video=output,
            dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    if dry_run:
        logger.info("\nDry run complete.")
        return 0

    logger.info("\nPipeline complete!")
    logger.info(f"Model: {results['model']}")
    logger.info(f"Demo video: {results['video']}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VIPE Gaussian Splatting")
    parser.add_argument("dataset", help="Path to input dataset")
    parser.add_argument("--workspace", "-w", default="workspace", help="Workspace directory")
    parser.add_argument("--output", "-o", default="demo.mp4", help="Output video")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without doing any work",
    )
    args = parser.parse_args()

    raise SystemExit(main(args.dataset, args.workspace, args.output, args.dry_run))
