from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import requests
import copy
import torch
from torch import nn
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import gc
import ncps
from ncps import wirings
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import pytorch_lightning as pl
import torch.utils.data as data
from transformers import BertModel, BertTokenizer
import pandas as pd
from torchvision import transforms
import torch.optim as optim
import glob
from spaital_mode import SpatialAwareMLP
from video_mask_processor import VideoMaskProcessor
from ultralytics import Drone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DroneControlSystem(nn.Module):
    def __init__(self, base_model = "./yolo11s-seg.pt", yolo_layer = 16, bert_output = None, seed = 22222, device = 'cpu'):
        super(DroneControlSystem, self).__init__()
        self.device = device
        self.bert_output = bert_output
        self.bert_padding = torch.zeros(1, 768).to(device) #Use padding if user don't wanna use text command
        self.mask_padding = torch.zeros(512).to(device)

        self.input_dim, self.spatial_height ,self.spatial_width = 128, 60, 80  #YOlo output from hook
        self.reduced_dim = 32   #channels after yolo mlp 32 --> 200,000 , 16 --> 100,000

        self.yolo_output = self.reduced_dim * self.spatial_height * self.spatial_width
        
        self.spatial_mlp = SpatialAwareMLP(
            input_dim=self.input_dim,       # Input channels from YOLO
            reduced_dim=self.reduced_dim,      # Reduced dimension (you can adjust this)
            spatial_height=self.spatial_height,   # Feature map height 
            spatial_width=self.spatial_width     # Feature map width
        ).to(device)

        self.processor = VideoMaskProcessor(in_channels=128, device=device)
        if bert_output is not None:
            self.processor.forward_bert_embedding(bert_output)

        self.yolo = Drone(base_model)
        self.yolo.eval()

        self.modified_output_from_hook = None
        self.hook_handle = self.yolo.model.model[yolo_layer].register_forward_hook(self.hook_function)

        self.wiring = wirings.NCP(
            inter_neurons=18,
            command_neurons=12,
            motor_neurons=4,
            sensory_fanout=6,
            inter_fanout=4,
            recurrent_command_synapses=4,
            motor_fanin=6,
            seed=seed,
        )

        bert_size = 768
        mask_size = 512
        self.input_size =  self.yolo_output + mask_size + bert_size 

        self.wiring.set_input_dim(self.input_size)
        self.wiring.build(self.input_size)
        self.rnn_cell = LTC(self.input_size, self.wiring, batch_first=True).to(self.device)
        self.current_state = None

    def forward(self, image, bert_output=None, point= None, training=False, new_sequence = False):
        if new_sequence:
            self.new_sequence()

        self.modified_output_from_hook = None
        with torch.no_grad():
            yolo_results = self.yolo(image)  # YOLO forward pass
        
        if len(yolo_results) != self.modified_output_from_hook.shape[0]:
            self.modified_output_from_hook = self.modified_output_from_hook[1:]

        mask = None
        for idx, yolo_result in enumerate(yolo_results):
            masks = None
            scaled_preds = None
            if yolo_result.masks is not None:
                masks = yolo_result.masks.data
                scaled_preds = yolo_result.scaled_preds.data
            feature = self.modified_output_from_hook[idx]
            if point is not None and idx == 0:  #After training change back that it can be recieved any time.
                self.processor_result = self.processor(feature, masks, scaled_preds, x_y_coordinates = point, training = training)
            else:
                self.processor_result = self.processor(feature, masks, scaled_preds, training=training)
            mask_to_concat = self.processor_result["mask"] if self.processor_result["mask"] is not None else self.mask_padding
            if mask is None:
                mask = mask_to_concat.unsqueeze(0)
            else:
                mask = torch.cat((mask, mask_to_concat.unsqueeze(0)), dim=0)
        
        processed_features = self.spatial_mlp(self.modified_output_from_hook.clone())
        flattened_features = processed_features.contiguous().view(processed_features.shape[0], -1)
           
        if bert_output is not None:            
            self.bert_output = bert_output
            self.processor.forward_bert_embedding(bert_output)
        if self.bert_output is None:
            bert_output = self.bert_padding
        else:
            bert_output = self.bert_output
        
        batch_size = flattened_features.shape[0]
        bert_output = bert_output.repeat(batch_size, 1)

        combined_output = torch.cat((flattened_features, mask, bert_output), dim=1) #bert_output_expanded
        combined_output = combined_output.unsqueeze(0)
        motor_output, self.current_state = self.rnn_cell(combined_output, self.current_state)
        
        gc.collect()
        torch.cuda.empty_cache()
        return {"motor_output": motor_output, 
                "visual_embed": processed_features, 
                "yolo": yolo_results ,
                **self.processor_result}

    # Hook function to capture the output of C3k2 Mid
    
    def hook_function(self, module, input, output):
        if output.shape[2] == 60:  # during 1st iteration shape is not 60 make sure it is 60.
            if self.modified_output_from_hook is None:
                self.modified_output_from_hook = output
            else:
                self.modified_output_from_hook = torch.cat((self.modified_output_from_hook, output), dim=0) # Concatenate along the batch dimension (dim=0)


    # Mind do not use this function NEVER in good sense
    def remove_hook(self):
        self.hook_handle.remove()

    def load_from_path(self, model_path):
        """Load model from a saved checkpoint"""
        checkpoint = torch.load(f'{model_path}/ltccell_weights.pt')
        self.wiring = checkpoint['wiring']
        rnn_cell_state_dict = checkpoint['state']
        self.current_state = checkpoint['current_state']
        self.input_size = checkpoint['input_size']
        
        self.rnn_cell = LTC(self.input_size, self.wiring, batch_first=True).to(device)
        self.rnn_cell.load_state_dict(rnn_cell_state_dict)

    def new_sequence(self):
        self.processor.new_sequence()
        self.current_state = None
        self.bert_output = torch.zeros(1, 768).to(device)

    def save_model(self, path):
        torch.save({'state': self.rnn_cell.state_dict(), 
                    'wiring': self.rnn_cell._wiring, 
                    'input_size':self.rnn_cell.input_size, 
                    'current_state': self.current_state},
                    f'{path}/yolo_model_lnn.pt')  #162 MB

    def reset_memory(self): #USe after each batch
        self.processor.reset_memory()
        self.current_state = self.current_state.detach().clone()
