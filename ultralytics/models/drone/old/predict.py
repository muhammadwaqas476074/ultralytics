# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, checks
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks

from .utils import adjust_bboxes_to_image_border

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

import cv2

class DronePredictor(SegmentationPredictor):    
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing for single-
    class segmentation.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, bert_model_name="bert-base-uncased", mlp_hidden_dim=768, mlp_output_dim=256):
        super().__init__(cfg, overrides, _callbacks)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(self.device)
        self.prompts = {}
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.bert_model.config.hidden_size, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim,mlp_output_dim)
        ).to(self.device)
        self.clip_model, self.clip_preprocess = None, None
        self.embedding_dim = mlp_output_dim
    
    

    #def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    #    """Initializes a FastSAMPredictor for fast SAM segmentation tasks in Ultralytics YOLO framework."""
    #    super().__init__(cfg, overrides, _callbacks)
    #    self.prompts = {}
    
    
    def get_object_embeddings_model(self, image, masks, model):
        """Extracts object embeddings based on masks."""
        object_embeddings = []
        for mask in masks:
            # Convert mask to boolean array
            mask = mask.astype(bool)

            # Apply mask to image to get object region
            object_region = image * mask[..., None]

            # Convert to PIL Image for embedding model input
            object_image = Image.fromarray(object_region.astype(np.uint8))
            object_image = object_image.resize((224, 224)) # Resize if necessary

            # Convert to tensor and move to device
            object_image = torch.from_numpy(np.array(object_image)).permute(2, 0, 1).float().to(self.device)
            object_image = object_image.unsqueeze(0) / 255.0

            # Get object embedding using embedding model
            with torch.no_grad():
                object_embedding = model.model.model[-1](object_image)
            object_embeddings.append(object_embedding.cpu().numpy())
        return np.array(object_embeddings)
    
    def get_object_embeddings(self, masks, device):  # Added device argument
        """Resizes masks and returns them as embeddings."""
        object_embeddings = []
        resized_size = 16  # Your desired mask size
        for mask in masks:
            resized_mask = self.resize_mask(mask, resized_size)
            # Convert to tensor, add channel dimension, and move to device
            resized_mask_tensor = torch.from_numpy(resized_mask).float().to(device) / 255.0 # Normalize 0-1
            resized_mask_tensor = resized_mask_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
            object_embeddings.append(resized_mask_tensor)

        return torch.stack(object_embeddings)
    
    def resize_mask(self, mask, size):
        """Resizes a mask to the specified size."""
        mask = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        return resized_mask
    

    def get_text_embedding(self, text):
        """Generates embedding for a text prompt."""
        text_input = self.bert_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            bert_output = self.bert_model(**text_input).last_hidden_state.mean(dim=1)
            text_embedding = self.mlp(bert_output)
        return text_embedding.detach().cpu().numpy()
    

    
    def postprocess(self, preds, img, orig_imgs):
        """Applies box postprocess for FastSAM predictions."""
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            results (Results | List[Results]): The original inference results from FastSAM models without any prompts.
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            texts (str | List[str], optional): Textual prompts, a list contains string objects.

        Returns:
            (List[Results]): The output results determined by prompts.
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # bboxes prompt
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(
                    points
                ), f"Excepted `labels` got same size as `point`, but got {len(labels)} and {len(points)}"
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def set_prompts(self, prompts):
        """Set prompts in advance."""
        self.prompts = prompts
