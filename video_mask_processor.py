import torch
import torch.nn.functional as F
import torch.nn as nn
from filterpy.kalman import KalmanFilter
import torchvision.ops as ops
import numpy as np
    
class KalmanBoxTracker:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = np.eye(4, 8)
        self.kf.R *= 10
        self.kf.P *= 100
        self.kf.Q *= 0.01
        self.kf.x[:4] = np.array(bbox).reshape(4, 1)
        self.age = 0
        self.last_update = 0
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        return self.kf.x[:4].flatten()

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape(4, 1))
        self.last_update = self.age

class VideoMaskProcessor(nn.Module):
    def __init__(self, in_channels, reference_mask=None, reference_bboxes=None, 
                 bert_embedding=None, target_size=(16, 16), device="cpu"):
        super().__init__()
        self.device = device
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # Reduce further
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # Final reduction
            nn.ReLU()).to(device)
        
        self.dense_layers = nn.Sequential(
            nn.Linear(16 * 16 * 16, 768),  # Directly project to embedding size
            nn.ReLU(),
            nn.Linear(768, 512)).to(device) # Final 512-dim embedding

        self.bert_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512)
        ).to(device)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_size).to(device)

        self.bert_embedding = bert_embedding  #During Inferance It should be different and bert projection will be stored after passingthorugh sequantial layer and used over and over
        if self.bert_embedding is not None:
            self.bert_embedding = torch.tensor(bert_embedding, device=device).float()
            self.bert_proj = self.bert_mlp(bert_embedding)
        else:
            self.bert_proj = None
        
        self.reference_mask = reference_mask
        self.reference_memory = []
        self.max_memory_size = 10  
        self.reference_memory_original = [] #for training 

        self.recent_features = [] 
        self.recent_features_original = [] #for training
        self.recent_features_size = 5      #While in Training this should match the batch size so that all memory is new before back propagation

        self.target_size = target_size
        self.tracker = None
        self.is_occluded = False
        self.occlusion_counter = 0
        self.current_radius = 0.5
        self.radius_step = 0.5
        self.reference_bboxes = reference_bboxes
        
        self.occlusion_threshold = 0.05  
        self.occlusion_frames = 5  
        self.base_reid_threshold = 0.90  
        self.max_reid_threshold = 0.98  
        self.max_memory_similarity_threshold = 0.80  
        self.min_memory_similarity_threshold = 0.60
        
        self.detection_history = []  
        
        self.in_detection_phase = False
        self.detection_phase_counter = 0
        self.detection_phase_required_frames = 10
        self.memory_update_frozen = False
        self.last_detected_feature = None
        self.candidate_detection_scores = []
        
        self.permanent_memory = [] 
        self.next_permanent_index = 0
        self.permanent_memory_threshold = 0.85
        self.use_temporary_in_occlusion = True

        self.triplet_margin = 0.2  # Add margin parameter for triplet loss
        self.triplet_loss = nn.TripletMarginLoss(margin=self.triplet_margin) # Initialize triplet_loss function <---- ADDED THIS LINE
        self.bert_proj = None  # Will be set through forward passes

    def forward_bert_embedding(self, bert_embedding):
        self.bert_embedding = bert_embedding.clone().detach().to(self.device).float()
        self.bert_proj = self.bert_mlp(self.bert_embedding)
        return self.bert_proj
    
    def remove_bert_embedding(self):
        self.bert_embedding = None
        self.bert_proj = None
    
    def add_reference_to_permanent(self, feature):
        """Modified to preserve gradients during training"""
        if feature.dim() == 1:
            # Already processed (512-dim), add as is
            processed = feature.clone()
        elif feature.dim() == 2:
            # Processed feature with batch dim, remove batch
            processed = feature.squeeze(0).clone()
        else:
            # Raw feature (3D/4D), process through reduction
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)  # Add batch dim
            if self.training:
                processed = self.reduce_channels(feature).squeeze(0)
            else:
                with torch.no_grad():
                    processed = self.reduce_channels(feature).squeeze(0)

        
        new_entry = {'feature': processed.clone(), 'index': self.next_permanent_index}
        self.permanent_memory.append(new_entry)
        self.next_permanent_index += 1
        return new_entry['index']
    
    def remove_from_permanent_by_index(self, index):
        self.permanent_memory = [entry for entry in self.permanent_memory if entry['index'] != index]

    def get_permanent_memory_info(self):
        return [{'index': entry['index']} for entry in self.permanent_memory]
    
    def reduce_channels(self, x):
        N, C, H, W = x.shape
        x = self.conv_layers(x)  # Apply convolutional layers
        x = x.view(N, -1)  # Flatten for MLP
        x = self.dense_layers(x)  # Pass through dense layers
        return x
    
    def start_training(self):
        self.training = True

    def stop_training(self):
        self.training = False

    def forward(self, features, masks, scaled_bboxes, alpha=0.3, 
                x_y_coordinates=None, remove_from_permanent_indices=None, 
                starting_idx=None, training=False, bert_embedding=None):
        if scaled_bboxes is None:
            output = self._handle_no_detections()
            return output

        self.training = training
        if bert_embedding is not None:
            self.bert_embedding = torch.tensor(bert_embedding, device=self.device).float()
            self.bert_proj = self.bert_mlp(self.bert_embedding)

        C, H_f, W_f = features.shape
        N, H_m, W_m = masks.shape
        
        processed_regions = torch.zeros((N, C, self.target_size[0], self.target_size[1]), device=self.device)

        for mask_idx in range(N):
            x_min, y_min, x_max, y_max = scaled_bboxes[mask_idx].tolist()
            if x_min >= x_max or y_min >= y_max:
                continue  
            
            pad_x = max(0, int((x_max - x_min) * 0.1))
            pad_y = max(0, int((y_max - y_min) * 0.1))
            
            x_min_pad = max(0, int(x_min) - pad_x)
            y_min_pad = max(0, int(y_min) - pad_y)
            x_max_pad = min(W_f, int(x_max) + pad_x)
            y_max_pad = min(H_f, int(y_max) + pad_y)
            
            cropped_features = features[:, y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            cropped_height, cropped_width = cropped_features.shape[1:]
            
            if cropped_height < 1 or cropped_width < 1:
                continue

            if cropped_height < 5 and cropped_width < 5:
                resized_masked_feat = self.adaptive_pool(cropped_features)
            else:
                mask = masks[mask_idx].unsqueeze(0).unsqueeze(0)
                scaled_mask = F.interpolate(mask.float(), size=(H_f, W_f), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                cropped_mask = scaled_mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                
                masked_feat = cropped_features * cropped_mask.unsqueeze(0).unsqueeze(0)
                if masked_feat.sum() == 0:  
                    masked_feat = cropped_features
                
                resized_masked_feat = self.adaptive_pool(masked_feat)

            processed_regions[mask_idx] = resized_masked_feat

        processed_reduced = self.reduce_channels(processed_regions)

        # Store processed_reduced for loss calculation
        self.last_processed = processed_reduced

        removed_permanent_indices = []
        if remove_from_permanent_indices is not None:
            self.permanent_memory = [entry for entry in self.permanent_memory 
                                     if entry['index'] not in remove_from_permanent_indices]
            removed_permanent_indices = remove_from_permanent_indices

        added_permanent_indices = []
        idx = None
        if x_y_coordinates is not None:
            if isinstance(x_y_coordinates, torch.Tensor):
                x, y = x_y_coordinates.tolist()
            if isinstance(x_y_coordinates, tuple):
                x, y = x_y_coordinates
            for idx, bbox in enumerate(scaled_bboxes):
                x_min, y_min, x_max, y_max = bbox.tolist()
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    perm_index = self.add_reference_to_permanent(processed_reduced[idx].unsqueeze(0))
                    added_permanent_indices.append(perm_index)
                    self.reference_mask = processed_reduced[idx].clone()
                    self.reference_mask_original = processed_regions[idx]

                    self.tracker = KalmanBoxTracker(bbox.tolist())
                    self.update_memory(self.reference_mask.clone(), self.reference_mask_original)
                    if self.training: #Store original feature to re Pass through network to get updated embeddings to calculate loss after batch
                        self.permanent_memory[-1]['original_features'] = processed_regions[idx].clone() #added storage.
                    break
        
        if starting_idx is not None:
            self.reference_mask = processed_reduced[starting_idx].clone()
            self.reference_mask_original = processed_regions[idx]
            self.tracker = KalmanBoxTracker(scaled_bboxes[starting_idx].tolist())
            self.update_memory(self.reference_mask.clone(), self.reference_mask_original)
            output = self._create_output(starting_idx, scaled_bboxes, 1.0)
            output.update({
                "added_permanent_indices": added_permanent_indices,
                "removed_permanent_indices": removed_permanent_indices,
                "idx":  idx
            })
            if self.training:
                output['loss'] = self.compute_training_loss(output)
            return output

        if self.tracker is None:
            best_perm_idx, best_perm_sim = self._check_permanent_memory(processed_reduced)
            if best_perm_idx is not None:
                self._start_tracking(best_perm_idx, scaled_bboxes, processed_reduced, processed_regions[best_perm_idx])
                output = self._create_output(best_perm_idx, scaled_bboxes, best_perm_sim)
                output.update({
                    "added_permanent_indices": added_permanent_indices,
                    "removed_permanent_indices": removed_permanent_indices,
                    "idx": best_perm_idx
                })
                if self.training:
                    output['loss'] = self.compute_training_loss(output)
                return output

            best_bert_idx, best_bert_sim = self._check_bert_embedding(processed_reduced)
            if best_bert_idx is not None:
                self._start_tracking(best_bert_idx, scaled_bboxes, processed_reduced, processed_regions[best_perm_idx])
                output = self._create_output(best_bert_idx, scaled_bboxes, best_bert_sim)
                output.update({
                    "added_permanent_indices": added_permanent_indices,
                    "removed_permanent_indices": removed_permanent_indices,
                    "idx": best_bert_idx
                })
                if self.training:
                    output['loss'] = self.compute_training_loss(output)
                return output

            output = self._create_output(None, scaled_bboxes, None)
            output.update({
                "added_permanent_indices": added_permanent_indices,
                "removed_permanent_indices": removed_permanent_indices,
                "idx": None
            })
            if self.training:
                output['loss'] = self.compute_training_loss(output)
            return output

        if self.is_occluded:
            output = self._handle_full_occlusion(processed_reduced, scaled_bboxes, processed_regions)
            output.update({
                "added_permanent_indices": added_permanent_indices,
                "removed_permanent_indices": removed_permanent_indices,
                "idx":  idx
            })
            if self.training:
                output['loss'] = self.compute_training_loss(output)
            return output

        predicted_bbox = torch.tensor(self.tracker.predict(), device=self.device)
        ious = ops.box_iou(predicted_bbox.unsqueeze(0), scaled_bboxes)[0]
        
        if torch.max(ious) < self.occlusion_threshold:
            self.memory_update_frozen = True
            output = self._handle_semi_occlusion(predicted_bbox, processed_reduced, scaled_bboxes)
            output.update({
                "added_permanent_indices": added_permanent_indices,
                "removed_permanent_indices": removed_permanent_indices,
                "idx":  idx
            })
            if self.training:
                output['loss'] = self.compute_training_loss(output)
            return output
        
        if self.in_detection_phase:
            output = self._continue_detection_phase(processed_reduced, scaled_bboxes, ious, alpha, processed_regions)
            output.update({
                "added_permanent_indices": added_permanent_indices,
                "removed_permanent_indices": removed_permanent_indices,
                "idx":  idx
            })
            if self.training:
                output['loss'] = self.compute_training_loss(output)
            return output
        
        output = self._normal_tracking(processed_reduced, scaled_bboxes, ious, alpha, processed_regions)
        output.update({
            "added_permanent_indices": added_permanent_indices,
            "removed_permanent_indices": removed_permanent_indices,
            "idx":  idx
        })
        if self.training:
            output['loss'] = self.compute_training_loss(output)
        print(output['mask'])
        return output
    
    def _handle_no_detections(self):

        if self.occlusion_counter >= self.occlusion_frames:
            self.is_occluded = True

        self.occlusion_counter +=1
        output = {
            "bboxes": None,
            "most_similar_idx": None,
            "similarity_score": None,
            "is_occluded": self.is_occluded,
            "occlusion_counter": self.occlusion_counter,
            "in_detection_phase": self.in_detection_phase,
            "detection_frame_count": self.detection_phase_counter if self.in_detection_phase else 0,
            "mask": None,
            "predicted_bbox": self.tracker.predict().flatten().tolist() if self.tracker else None
        }
        return output

    def _check_permanent_memory(self, processed_reduced):
        if not self.permanent_memory:
            return None, None
        processed_flat = processed_reduced.view(len(processed_reduced), -1)
        max_sim = -1
        best_idx = -1
        best_permanent_id = None
        for entry in self.permanent_memory:
            perm_feature = entry['feature'].flatten().unsqueeze(0)
            similarities = F.cosine_similarity(processed_flat, perm_feature)
            scaled_sim = (similarities + 1) / 2
            current_max, current_idx = torch.max(scaled_sim, 0)
            if current_max > self.permanent_memory_threshold and current_max > max_sim:
                max_sim = current_max
                best_idx = current_idx.item()
                best_permanent_id = entry['index']
        if max_sim >= self.permanent_memory_threshold:
            return best_idx, max_sim
        return None, None

    def _check_bert_embedding(self, processed_reduced):
        if self.bert_proj is None:
            return None, None
        bert_proj = self.bert_proj.flatten().unsqueeze(0)
        processed_flat = processed_reduced.view(len(processed_reduced), -1)
        similarities = F.cosine_similarity(processed_flat, bert_proj)
        scaled_sim = (similarities + 1) / 2
        max_sim, best_idx = torch.max(scaled_sim, 0)
        if max_sim >= self.base_reid_threshold:
            return best_idx.item(), max_sim.item()
        return None, None

    def _start_tracking(self, idx, scaled_bboxes, processed_reduced, processed_region):
        self.reference_mask = processed_reduced[idx].clone()
        
        self.tracker = KalmanBoxTracker(scaled_bboxes[idx].tolist())
        self.update_memory(self.reference_mask.clone(), processed_region)
        self.update_recent_features(self.reference_mask.clone())
        center = ((scaled_bboxes[idx][:2] + scaled_bboxes[idx][2:]) / 2)
        self.detection_history.append(center.cpu().numpy())
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)

    def _handle_full_occlusion(self, processed_reduced, scaled_bboxes, processed_regions):
        self.memory_update_frozen = True
        best_perm_idx, best_perm_sim = self._check_permanent_memory(processed_reduced)
        if best_perm_idx is not None:
            self._start_tracking(best_perm_idx, scaled_bboxes, processed_reduced, processed_regions[best_perm_idx])
            output = self._create_output(best_perm_idx, scaled_bboxes, best_perm_sim, in_detection=True)
            output['permanent_memory_match'] = best_perm_idx
            return output

        best_bert_idx, best_bert_sim = self._check_bert_embedding(processed_reduced)
        if best_bert_idx is not None:
            self._start_tracking(best_bert_idx, scaled_bboxes, processed_reduced, processed_regions[best_perm_idx])
            output = self._create_output(best_bert_idx, scaled_bboxes, best_bert_sim, in_detection=True)
            return output

        processed_flat = processed_reduced.view(len(processed_reduced), -1)
        max_sim = torch.tensor(-1.0, device=self.device)
        best_idx = -1
        
        if self.recent_features:
            for ref_feat in self.recent_features:
                similarities = F.cosine_similarity(processed_flat, ref_feat.flatten().unsqueeze(0))
                current_max, current_idx = torch.max(similarities, 0)
                scaled_sim = (current_max + 1) / 2
                if scaled_sim > max_sim:
                    max_sim = scaled_sim
                    best_idx = current_idx
        
        for mem_ref in self.reference_memory:
            similarities = F.cosine_similarity(processed_flat, mem_ref.flatten().unsqueeze(0))
            current_max, current_idx = torch.max(similarities, 0)
            scaled_sim = (current_max + 1) / 2
            if scaled_sim > max_sim:
                max_sim = scaled_sim
                best_idx = current_idx

        if self.detection_history and best_idx >= 0:
            candidate_bbox = scaled_bboxes[best_idx]
            candidate_center = ((candidate_bbox[:2] + candidate_bbox[2:]) / 2).cpu().numpy()
            last_center = self.detection_history[-1]
            distance = np.linalg.norm(candidate_center - last_center)
            motion_weight = 0.9 ** min(10, distance)
            max_sim = max_sim * motion_weight

        scaled_sim = (max_sim + 1) / 2 if max_sim < 0 else max_sim
        
        if scaled_sim >= self.base_reid_threshold:
            self._start_detection_phase(scaled_bboxes[best_idx], processed_reduced[best_idx], scaled_sim)
            self.tracker = KalmanBoxTracker(scaled_bboxes[best_idx].tolist())
            output = self._create_output(best_idx, scaled_bboxes, scaled_sim, in_detection=True)
            return output
        
        output = self._create_output(None, scaled_bboxes, None)
        return output

    def _handle_semi_occlusion(self, predicted_bbox, processed_reduced, scaled_bboxes):
        # Set memory_update_frozen to True for semi occlusion
        self.memory_update_frozen = True
        
        self.occlusion_counter += 1
        progress = min(self.occlusion_counter / self.occlusion_frames, 1.0)
        current_threshold = max(0.5, self.base_reid_threshold - 0.05 * progress)
        
        if self.occlusion_counter == 1:
            self._initialize_radius()
        
        self.current_radius += self.radius_step * (1 + progress)

        pred_center = (predicted_bbox[:2] + predicted_bbox[2:]) / 2
        bbox_centers = (scaled_bboxes[:, :2] + scaled_bboxes[:, 2:]) / 2
        distances = torch.norm(bbox_centers - pred_center, dim=1)
        in_radius = distances < self.current_radius
        
        if not in_radius.any():
            if self.occlusion_counter >= self.occlusion_frames:
                self.is_occluded = True
            output = self._create_output(None, scaled_bboxes, None)
            return output

        candidates = processed_reduced[in_radius]
        candidate_indices = torch.where(in_radius)[0]
        candidates_flat = candidates.view(candidates.size(0), -1)
        ref_flat = self.reference_mask.flatten().unsqueeze(0)
        primary_similarities = (F.cosine_similarity(candidates_flat, ref_flat, dim=1) + 1) / 2
        
        if self.recent_features:
            temp_similarities = torch.zeros_like(primary_similarities)
            for i, recent_feat in enumerate(self.recent_features):
                recent_flat = recent_feat.flatten().unsqueeze(0)
                recency_weight = 0.8 ** (len(self.recent_features) - i - 1)
                temp_sim = (F.cosine_similarity(candidates_flat, recent_flat, dim=1) + 1) / 2
                temp_similarities += temp_sim * recency_weight
            
            if len(self.recent_features) > 0:
                temp_similarities /= len(self.recent_features)
                primary_similarities = 0.7 * primary_similarities + 0.3 * temp_similarities   #Waitage given to each similarity
        
        # Only consider detections with similarity above base threshold
        valid = primary_similarities >= self.base_reid_threshold
        
        if valid.any():
            best_idx = torch.argmax(primary_similarities)
            original_idx = candidate_indices[best_idx]
            similarity_score = primary_similarities[best_idx]
            
            # Start detection phase instead of immediately accepting
            self._start_detection_phase(scaled_bboxes[original_idx], processed_reduced[original_idx], similarity_score)
            
            output = self._create_output(original_idx, scaled_bboxes, similarity_score, in_detection=True)
            return output
            
        output = self._create_output(None, scaled_bboxes, None)
        return output

    def _initialize_radius(self):
        if self.reference_bboxes is not None:
            w = self.reference_bboxes[2] - self.reference_bboxes[0]
            h = self.reference_bboxes[3] - self.reference_bboxes[1]
            self.current_radius = torch.sqrt(w**2 + h**2) / 2
            self.radius_step = max(1.0, (w + h) / 8)
        else:
            self.current_radius = 20.0  
            self.radius_step = 5.0
    
    def _start_detection_phase(self, bbox, feature, similarity):
        """Start the detection phase after occlusion"""
        self.in_detection_phase = True
        self.detection_phase_counter = 1  # First frame
        self.occlusion_counter = 0
        self.is_occluded = False
        self.last_detected_feature = feature.clone()
        self.candidate_detection_scores = [similarity]
        
        # Update tracking info but don't update memory yet
        self.reference_mask = feature.clone()
        self.reference_bboxes = bbox
        
        # Add to detection history for motion tracking
        center = ((bbox[:2] + bbox[2:]) / 2).cpu().numpy()
        self.detection_history.append(center)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)

    def _continue_detection_phase(self, processed_reduced, scaled_bboxes, ious, alpha, processed_regions):
        """Continue detection phase to verify object over multiple frames"""
        # Use Kalman filter to predict location
        predicted_bbox = torch.tensor(self.tracker.predict(), device=self.device)
        ious = ops.box_iou(predicted_bbox.unsqueeze(0), scaled_bboxes)[0]
        
        # Calculate feature similarity
        processed_flat = processed_reduced.view(len(processed_reduced), -1)
        ref_flat = self.reference_mask.flatten().unsqueeze(0)
        cosine_sim = (F.cosine_similarity(processed_flat, ref_flat) + 1) / 2
        
        # Motion weighting
        motion_weight = torch.ones_like(ious)
        if self.detection_history:
            last_center = torch.tensor(self.detection_history[-1], device=self.device)
            current_centers = (scaled_bboxes[:, :2] + scaled_bboxes[:, 2:]) / 2
            distances = torch.norm(current_centers - last_center, dim=1)
            motion_weight = torch.exp(-distances / 50)
        
        # Combined score
        weighted_sim = alpha * cosine_sim + (1 - alpha) * 0.7 * ious + (1 - alpha) * 0.3 * motion_weight
        best_idx = torch.argmax(weighted_sim).item()
        
        # Update Kalman filter
        self.tracker.update(scaled_bboxes[best_idx].cpu().numpy())
        
        # Check if similarity is still above threshold
        if cosine_sim[best_idx] >= self.base_reid_threshold:
            # Increment detection counter
            self.detection_phase_counter += 1
            
            # Update reference mask for next comparison
            self.reference_mask = processed_reduced[best_idx].clone()
            self.reference_mask_original = processed_regions[best_idx].clone()
            self.reference_bboxes = scaled_bboxes[best_idx]
            
            # Add to detection history
            center = ((scaled_bboxes[best_idx][:2] + scaled_bboxes[best_idx][2:]) / 2).cpu().numpy()
            self.detection_history.append(center)
            if len(self.detection_history) > 10:
                self.detection_history.pop(0)
            
            # Record similarity score
            self.candidate_detection_scores.append(cosine_sim[best_idx].item())
            
            # Check if we have enough consistent frames
            if self.detection_phase_counter >= self.detection_phase_required_frames:
                # Successful detection phase completed
                self._exit_detection_phase(scaled_bboxes[best_idx], processed_reduced[best_idx])
            
            output = self._create_output(best_idx, scaled_bboxes, cosine_sim[best_idx], in_detection=True)
            return output
        else:
            # Reset detection phase if similarity drops below threshold
            self.in_detection_phase = False
            self.detection_phase_counter = 0
            self.candidate_detection_scores = []
            
            # Keep memory update frozen as we're still uncertain
            output = self._create_output(None, scaled_bboxes, None)
            return output

    def _exit_detection_phase(self, bbox, feature):
        """Exit detection phase after successful verification"""
        self.in_detection_phase = False
        self.detection_phase_counter = 0
        self.memory_update_frozen = False  # Now we can update memory again
        
        # Update memory if appropriate
        avg_score = sum(self.candidate_detection_scores) / len(self.candidate_detection_scores)
        self.update_memory(feature.clone())
        self.update_recent_features(feature.clone())
        
        # Reset candidate scores
        self.candidate_detection_scores = []

    def _normal_tracking(self, processed_reduced, scaled_bboxes, ious, alpha, processed_regions):
        processed_flat = processed_reduced.view(len(processed_reduced), -1)
        ref_flat = self.reference_mask.flatten().unsqueeze(0)
        cosine_sim = (F.cosine_similarity(processed_flat, ref_flat) + 1) / 2
        
        motion_weight = torch.ones_like(ious)
        if self.detection_history:
            last_center = torch.tensor(self.detection_history[-1], device=self.device)
            current_centers = (scaled_bboxes[:, :2] + scaled_bboxes[:, 2:]) / 2
            distances = torch.norm(current_centers - last_center, dim=1)
            motion_weight = torch.exp(-distances / 50)
        
        weighted_sim = alpha * cosine_sim + (1 - alpha) * 0.7 * ious + (1 - alpha) * 0.3 * motion_weight
        best_idx = torch.argmax(weighted_sim).item()
        
        self.tracker.update(scaled_bboxes[best_idx].cpu().numpy())
        self.reference_mask = processed_reduced[best_idx].clone()
        self.reference_mask_original = processed_regions[best_idx].clone()
        self.reference_bboxes = scaled_bboxes[best_idx]
        center = ((scaled_bboxes[best_idx][:2] + scaled_bboxes[best_idx][2:]) / 2).cpu().numpy()
        self.detection_history.append(center)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        # In normal tracking, we can update features and memory
        if weighted_sim[best_idx] > 0.8 and not self.memory_update_frozen:
            self.update_recent_features(processed_reduced[best_idx].clone())
        
        output = self._create_output(best_idx, scaled_bboxes, weighted_sim[best_idx])
        return output

    def get_mask_from_reference(self, reference_mask, permanent_memory):
        """Retrieves mask from reference_mask or permanent_memory, prioritizing reference_mask."""
        if isinstance(reference_mask, list):
            if len(reference_mask) > 0 and reference_mask[0] is not None:
                return reference_mask[0]
            else:
                return self.get_mask_from_permanent_memory(permanent_memory)
        else:  # reference_mask is not a list
            if reference_mask is not None:
                return reference_mask
            else:
                return self.get_mask_from_permanent_memory(permanent_memory)
            
    def get_mask_from_permanent_memory(self, permanent_memory):
        """Retrieves the 'feature' from the first element of permanent_memory if valid."""
        if permanent_memory is not None and len(permanent_memory) > 0 and permanent_memory[0].get('feature') is not None:
            return permanent_memory[0]['feature']
        return None


    def _create_output(self, idx, scaled_bboxes, similarity, in_detection=False):
        similarity_value = similarity.item() if isinstance(similarity, torch.Tensor) else similarity
        idx_value = idx.item() if isinstance(idx, torch.Tensor) else idx

        if not self.is_occluded and self.reference_mask is not None:
            mask = self.get_mask_from_reference(self.reference_mask, self.permanent_memory)
        else:
            mask = self.get_mask_from_permanent_memory(self.permanent_memory)


        output = {
            "bboxes": scaled_bboxes[idx_value].tolist() if idx_value is not None else None,
            "most_similar_idx": idx_value,
            "similarity_score": similarity_value,
            "is_occluded": self.is_occluded,
            "occlusion_counter": self.occlusion_counter,
            "in_detection_phase": self.in_detection_phase,
            "detection_frame_count": self.detection_phase_counter if self.in_detection_phase else 0,
            "mask": mask
        }
        return output 

    def update_memory(self, candidate_feature, reference_mask_original):
        # Skip memory updates if frozen
        if self.memory_update_frozen:
            return
            
        candidate_feature_flat = candidate_feature.flatten()
        if not self.reference_memory:
            self.reference_memory.append(candidate_feature.clone())
            self.reference_memory_original.append(reference_mask_original.clone())
            return
        max_similarity = 0
        for mem_ref in self.reference_memory:
            mem_ref_flat = mem_ref.flatten()
            sim = F.cosine_similarity(candidate_feature_flat.unsqueeze(0), 
                                     mem_ref_flat.unsqueeze(0), dim=1)
            max_similarity = max(max_similarity, (sim + 1) / 2)
        if max_similarity < self.max_memory_similarity_threshold and max_similarity > self.min_memory_similarity_threshold:
            self.reference_memory.append(candidate_feature.clone())
            self.reference_memory_original.append(reference_mask_original.clone())
            if len(self.reference_memory) > self.max_memory_size:
                max_pair_sim = -1
                similar_pair = (0, 1)
                for i in range(len(self.reference_memory)):
                    for j in range(i+1, len(self.reference_memory)):
                        feat_i = self.reference_memory[i].flatten()
                        feat_j = self.reference_memory[j].flatten()
                        sim = F.cosine_similarity(feat_i.unsqueeze(0), 
                                                 feat_j.unsqueeze(0), dim=1)
                        sim = (sim + 1) / 2
                        if sim > max_pair_sim:
                            max_pair_sim = sim
                            similar_pair = (i, j)
                self.reference_memory.pop(similar_pair[0])
                self.reference_memory_original.pop(similar_pair[0])
            else:
                self.reference_memory.pop(-1)
                self.reference_memory_original.pop(-1)
    
    def update_recent_features(self, feature):
        # Skip recent features update if memory is frozen
        if self.memory_update_frozen:
            return
            
        self.recent_features.append(feature.clone())
        if len(self.recent_features) > self.recent_features_size:
            self.recent_features.pop(0)

    def compute_training_loss(self, output):
        """
        Compute diverse training losses based on detection and occlusion states.
        """
        losses = {}
    
        if output['most_similar_idx'] is not None:
            # Object detected
            anchor_idx = output['most_similar_idx']
            anchor = self.last_processed[anchor_idx]
    
            # Positives
            positives = [anchor]  # Start with the detected object
            if self.bert_proj is not None:
                positives.append(self.bert_proj)
            positives.extend([entry['feature'] for entry in self.permanent_memory])
            positives.extend(self.reference_memory)
            
            # Negatives
            mask = torch.ones(len(self.last_processed), dtype=torch.bool, device=self.device)
            mask[anchor_idx] = False
            negatives = self.last_processed[mask]
    
            # Multi-Positive Contrastive Loss with Hardest Negative Mining
            total_contrastive_loss = 0.0
            for pos in positives:
                pos_sim = F.cosine_similarity(anchor.unsqueeze(0), pos.unsqueeze(0))
                if len(negatives) > 0:
                    neg_sims = F.cosine_similarity(anchor.unsqueeze(0), negatives)
                    hardest_neg = torch.max(neg_sims)
                    loss_term = F.relu(-pos_sim + hardest_neg + self.triplet_margin)
                    if loss_term.dim() > 0:
                        loss_term = loss_term.mean()
                        
                    total_contrastive_loss += loss_term
    
            if len(positives) > 0:
                losses['contrastive_loss'] = total_contrastive_loss / len(positives)
            else:
                losses['contrastive_loss'] = torch.tensor(0.0, device=self.device)
    
            # Semantic Alignment (BERT)
            if self.bert_proj is not None:
                bert_loss = 0.0
                for pos in positives:
                    bert_loss += 1 - F.cosine_similarity(self.bert_proj.unsqueeze(0), pos.unsqueeze(0))
                losses['bert_alignment_loss'] = bert_loss.mean()
    
            # Triplet Loss with Permanent Memory as Anchors
            if len(self.permanent_memory) > 0:
                anchors_stack = torch.stack([m['feature'] for m in self.permanent_memory])
                permanent_triplet_loss = 0.0
                for anchor_idx_perm in range(anchors_stack.shape[0]):
                    anchor = anchors_stack[anchor_idx_perm]
                    positive_expanded = anchor.unsqueeze(0).expand(1, -1) # Corrected expansion

                    if len(negatives) > 0:
                        triplet_loss_item = self.triplet_loss(anchor.unsqueeze(0), positive_expanded, negatives)
                        permanent_triplet_loss += triplet_loss_item

                if anchors_stack.shape[0] > 0:
                    losses['permanent_triplet_loss'] = permanent_triplet_loss / anchors_stack.shape[0]
                else:
                    losses['permanent_triplet_loss'] = torch.tensor(0.0, device=self.device)

            # Feature Reconstruction Loss
            recon_loss = 0.0
            num_anchors = anchors_stack.shape[0]
            for i in range(num_anchors):
                recon_loss += F.mse_loss(anchors_stack[i], self.reference_mask)
            recon_loss = recon_loss / num_anchors
            losses['recon_loss'] = recon_loss
    
        else:
            # Occlusion (no object detected)
            references = []
            if self.bert_proj is not None:
                references.append(self.bert_proj)
            references.extend([entry['feature'] for entry in self.permanent_memory])
            references.extend(self.reference_memory)
    
            anti_similarity_loss = 0.0
            if len(references) > 0 and len(self.last_processed) > 0:
                for ref in references:
                    sims = F.cosine_similarity(ref.unsqueeze(0), self.last_processed)
                    anti_similarity_loss += torch.mean(F.relu(sims + 0.2))
                losses['occlusion_loss'] = anti_similarity_loss / len(references)
            else:
                losses['occlusion_loss'] = torch.tensor(0.0, device=self.device)
        return losses
    
    def new_sequence(self): #During trainign call before new sequence to empty old data
        """Reset all temporary states for new sequence"""
        self.reference_bboxes = None
        self.reference_memory = []
        self.reference_memory_original = []
        self.reference_mask = None
        self.reference_mask_original = None
        self.recent_features_original = []
        self.recent_features = []
        self.permanent_memory = []
        self.detection_history = []
        self.candidate_detection_scores = [] 
        self.tracker = None
        self.bert_embedding = None
        self.bert_proj = None
        self.is_occluded = False
        self.occlusion_counter = 0
    
    def reset_memory(self):  # During training update after each batch
    # Reference memory limit is same as batch size so every object is already updated.
    # self.recent_features = self.recent_features.detach

        if hasattr(self, 'reference_mask_original') and self.reference_mask_original is not None:
            print(self.reference_mask_original)
            print(type(self.reference_mask_original))
            print(self.reference_mask_original.shape)
            self.reference_mask = self.reduce_channels(self.reference_mask_original.unsqueeze(0)).squeeze(0)
    
        if hasattr(self, 'reference_bboxes') and self.reference_bboxes is not None:
            self.reference_bboxes = self.reference_bboxes.detach()
    
        if hasattr(self, 'bert_embedding') and self.bert_embedding is not None:
            self.bert_proj = self.bert_mlp(self.bert_embedding)
    
        if hasattr(self, 'permanent_memory') and self.permanent_memory is not None:
            for entry in self.permanent_memory:
                if 'original_features' in entry and entry['original_features'] is not None:  # Check if original_features exists and is not None
                    entry['feature'] = self.reduce_channels(entry['original_features'].unsqueeze(0)).squeeze(0)  # recalculate features.
    
        if hasattr(self, 'reference_memory_original') and self.reference_memory_original is not None:
            self.reference_memory = []
            for original_feature in self.reference_memory_original:
                if original_feature is not None: #check if original feature is not none
                    reprocessed_feature = self.reduce_channels(original_feature.unsqueeze(0)).squeeze(0)
                    self.reference_memory.append(reprocessed_feature.clone())