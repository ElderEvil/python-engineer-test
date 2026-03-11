"""Task 4: VIPE Gaussian Splatting Demo.

Set up VIPE, generate Gaussian Splatting reconstruction, create demo video.
"""

from .vipe_runner import VIPEManager, run_vipe_pipeline

__all__ = ["VIPEManager", "run_vipe_pipeline"]
