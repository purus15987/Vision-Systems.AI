"""
Utilities, Pipeline Automation, State History, and Hardware Detection Module.
Provides Undo/Redo history stack manager, JSON pipeline import/export,
and CUDA/OpenCL GPU acceleration detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import os
import numpy as np
import cv2


class HistoryManager:
    """Manages Undo/Redo image state history for interactive app sessions."""

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

    def push(self, img: np.ndarray):
        """Pushes a copy of current image state onto the undo stack."""
        self.undo_stack.append(img.copy())
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, current_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Reverts to the previous image state."""
        if not self.undo_stack:
            return current_img, False
        self.redo_stack.append(current_img.copy())
        prev_img = self.undo_stack.pop()
        return prev_img, True

    def redo(self, current_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Redoes the last undone operation."""
        if not self.redo_stack:
            return current_img, False
        self.undo_stack.append(current_img.copy())
        next_img = self.redo_stack.pop()
        return next_img, True

    def reset(self, initial_img: np.ndarray):
        """Resets the history stacks to a fresh base image."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.undo_stack.append(initial_img.copy())


class PipelineManager:
    """Serializes, deserializes, and executes sequential processing pipelines as JSON."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, operation: str, params: Dict[str, Any]):
        """Appends an operation step with parameters to the pipeline."""
        self.steps.append({
            "operation": operation,
            "params": params
        })

    def export_to_json(self, filepath: str) -> str:
        """Saves current processing pipeline to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"pipeline": self.steps}, f, indent=4)
        return filepath

    def load_from_json(self, filepath: str) -> List[Dict[str, Any]]:
        """Loads a processing pipeline from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.steps = data.get("pipeline", [])
        return self.steps


def detect_gpu_acceleration() -> Dict[str, Any]:
    """
    Detects OpenCV CUDA or OpenCL hardware acceleration support.
    Returns status dictionary.
    """
    info = {
        "cuda_available": False,
        "cuda_devices": 0,
        "opencl_available": cv2.ocl.haveOpenCL(),
        "opencl_enabled": cv2.ocl.useOpenCL()
    }

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        info["cuda_devices"] = count
        info["cuda_available"] = count > 0
    except Exception:
        pass

    return info
