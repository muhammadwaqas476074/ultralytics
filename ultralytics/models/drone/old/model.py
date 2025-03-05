# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model

from .predict import DronePredictor
from .val import FastSAMValidator
from .DroneModel import DroneModel
from .trainer import DroneTrainer

import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, checks
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class Drone(Model):
    """
    Drone model interface.

    Example:
        ```python
        from ultralytics import Drone

        model = Drone("last.pt")
        results = model.predict("ultralytics/assets/bus.jpg")
        ```
    """

    def __init__(self, model="yolo11n.pt", task="segment", verbose=False):
        """Call the __init__ method of the parent class (YOLO) with the updated default model."""
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)
    

    def predict(self, source, task_text = "Maintain your Position", stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs):
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        results = super().predict(source, stream, prompts=prompts, **kwargs)
        if not stream:
            for i in range(len(results)):
                masks = results[i].masks.data.cpu().numpy()
                image_embeddings = self.predictor.get_object_embeddings(masks, self.predictor.device) # Pass the device

                text_embedding = self.predictor.get_text_embedding(task_text)
                #image_embeddings = get_object_embeddings(results[i].orig_img, results[i].masks.data.cpu().numpy(), self.model)

                results[i].image_embeddings = image_embeddings
                results[i].text_embedding = text_embedding
                similarities = []
                print(len(image_embeddings))
                for image_embedding in image_embeddings:
                    # Flatten the mask for similarity calculation
                    print(image_embedding.shape)
                    print(text_embedding.shape)
                    similarities.append(cosine_similarity(image_embedding.cpu().flatten().reshape(1,-1), text_embedding)[0][0])
                results[i].similarities = similarities
                if similarities:
                    max_similarity_index = np.argmax(similarities)
                    results[i].most_similar_object = results[i].masks.data[max_similarity_index].cpu().numpy()
                else:
                    results[i].most_similar_object = None
        return results

    @property
    def task_map(self):
        return {"segment": {"model": DroneModel, "predictor": DronePredictor, "validator": FastSAMValidator, "trainer": DroneTrainer}}
