# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from .predict import SegmentationPredictor
from .train import SegmentationTrainer
from .val import SegmentationValidator
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import ROOT, yaml_load


class Drone(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "segment": {
                "model": SegmentationModel,
                "trainer": SegmentationTrainer,
                "validator": SegmentationValidator,
                "predictor": SegmentationPredictor,
            }
        }
