import text_processor
import torch
import torch.nn as nn
import torch.optim as optim
import os, glob
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torchvision import transforms
from text_processor import TextProcessor
from drone_model import DroneControlSystem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_and_cleanup_checkpoints(checkpoint, checkpoint_path, checkpoint_dir):
    """
    Saves a checkpoint and deletes old checkpoint files.

    Args:
        checkpoint (dict): The checkpoint dictionary to save.
        checkpoint_path (str): The full path to save the checkpoint.
        checkpoint_dir (str): The directory containing all checkpoints.
    """

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Delete old checkpoints
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    for cp in all_checkpoints:
        if cp != checkpoint_path:
            try:
                os.remove(cp)
                print(f"Deleted old checkpoint: {cp}")
            except Exception as e:
                print(f"Error deleting checkpoint {cp}: {e}")
 
class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu', bert_model=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.sequences = []
        self.sequence_lengths = []
        self.bert_model = bert_model
        self.samples = []

        for sequence_folder in sorted(os.listdir(root_dir)):
            sequence_path = os.path.join(root_dir, sequence_folder)

            if os.path.isdir(sequence_path):
                image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])

                if 'output.csv' in os.listdir(sequence_path):
                    self.sequences.append((sequence_path, image_files))
                    self.sequence_lengths.append(len(image_files))
                else:
                    print(f"output.csv not found in: {sequence_path}")
            else:
                print(f"{sequence_path} is not a directory")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_path, image_files = self.sequences[idx]
        images = []
        targets = []

        output_csv = pd.read_csv(os.path.join(sequence_path, "output.csv"))

        if len(image_files) > len(output_csv):
            image_files = image_files[:len(output_csv)]

        for image_file in image_files:
            image_path = os.path.join(sequence_path, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}") # DEBUG: Print image_path in error message
                continue

        for index, row in output_csv.iterrows():
            x_cord = row.get("x_cord")
            y_cord = row.get("y_cord")
            text = row.get("text")

            point = None
            if pd.notna(x_cord) and pd.notna(y_cord):
                point = (float(x_cord), float(y_cord))
                
            bert_embedding = None
            if isinstance(text, str) and pd.notna(text):
                try:
                    bert_embedding = self.bert_model(text.strip())
                except Exception as bert_e:
                    print(f"  Error creating BERT embedding for text '{text}': {bert_e}")
                    bert_embedding = None

            target = {
                "motor_targets": torch.tensor([row["roll"], row["pitch"], row["yaw"], row["throttle"]], dtype=torch.float32).to(self.device),
                "point": point,
                "bert": bert_embedding
            }
            
            targets.append(target)

        if len(images) > len(targets):
            images = images[:len(targets)]
        if len(targets) > len(images):
            targets = targets[:len(images)]

        if not images:
            raise ValueError(f"No images loaded for sequence at {sequence_path}. Check image file extensions and folder content.")
        if not targets:
            raise ValueError(f"No targets loaded for sequence at {sequence_path}. Check output.csv content.")

        return images, targets

def custom_collate_fn(batch):
    images_batch = [item[0] for item in batch] # item[0] is images_list, already list of tensors
    targets_batch = [item[1] for item in batch] # item[1] is targets_list, list of dicts

    # Collate targets - needs special handling for 'bert' which can be None
    collated_targets = []
    for targets_list in targets_batch: # Iterate through list of targets (for each sequence)
        collated_sequence_targets = []
        for target in targets_list: # Now process each target dict in a sequence
            bert_embedding = target['bert']
            
            collated_target = {
                'motor_targets': target['motor_targets'],
                'point': target['point'],
                'bert': bert_embedding # Now always a tensor for collation
            }
            collated_sequence_targets.append(collated_target)
        collated_targets.append(collated_sequence_targets) # List of list of target dicts, bert is tensor


    return images_batch, collated_targets # Return lists of images and collated targets

def train_model(model, dataset, epochs, optimizer, checkpoint_dir, device, batch_size=10, start_epoch=0):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))

    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']) #  ], strict = False
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'] )
        start_epoch = checkpoint['epoch']
        start_seq_idx = checkpoint.get('seq_idx', 0) #load the sequence index.
        print(f"Resuming from epoch {start_epoch}, sequence {start_seq_idx}")
    else:
        start_seq_idx = 0
        print(f"Starting from scratch, epoch {start_epoch}, sequence {start_seq_idx}")

    loss_weights = {
        'motor_loss': 1.0,
        'contrastive_loss': 0.5,
        'bert_alignment_loss': 0.3,
        'permanent_triplet_loss': 0.2,
        'recon_loss': 0.4,
        'occlusion_loss': 0.3
    }

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        total_loss = 0.0
        sequence_counter = 0

        for seq_idx, sequence_data in enumerate(loader):
            images_list, targets_list = sequence_data
            images = images_list[0]  # Extract images from the single sequence
            targets = targets_list[0] #Extract targets from single sequence
            seq_len = len(images)

            sequence_loss = 0.0
            i = 0
            while i < seq_len:
                start_index = i
                end_index = min(i + batch_size, seq_len)
                batch_imgs = images[start_index:end_index]
                batch_tgts = targets[start_index:end_index]
                batch_tensor = torch.stack(batch_imgs).to(device)
                motor_tgts = torch.stack([t['motor_targets'] for t in batch_tgts]).to(device)

                point_input = None
                bert_input = None

                point_input_list = [t['point'] for t in batch_tgts if t['point'] is not None]
                bert_input_list = [t['bert'] for t in batch_tgts if t['bert'] is not None]

                point_input = point_input_list[0] if point_input_list else None

                bert_input = bert_input_list[0] if bert_input_list else None

                optimizer.zero_grad()
                outputs = model(
                    batch_tensor,
                    point=point_input,
                    bert_output=bert_input,
                    training=True
                )

                motor_loss = F.mse_loss(outputs['motor_output'], motor_tgts)
                processor_losses = outputs.get('loss', {})

                total_batch_loss = motor_loss * loss_weights['motor_loss']
                for loss_name, loss_value in processor_losses.items():
                    if loss_value is not None:
                        total_batch_loss += loss_value * loss_weights.get(loss_name, 1.0)
    
                total_batch_loss.backward()
                optimizer.step()
                model.reset_memory()

                sequence_loss += total_batch_loss.item()

                del outputs, processor_losses, motor_loss
                torch.cuda.empty_cache()

                i = end_index

            total_loss += sequence_loss / (seq_len / batch_size)
            sequence_counter += 1
            model.new_sequence()
            avg_seq_loss = sequence_loss / (seq_len / batch_size)
            
            sequence_checkpoint = {
                'epoch': epoch + 1,
                'seq_idx': seq_idx + 1, #save the next sequence index.
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / sequence_counter
            }
            sequence_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_seq_{seq_idx+1}.pth')
            save_and_cleanup_checkpoints(sequence_checkpoint, sequence_checkpoint_path, checkpoint_dir)

        avg_epoch_loss = total_loss / sequence_counter
        epoch_time = time.time() - epoch_start
        time_remaining = (epochs - epoch - 1) * epoch_time

        epoch_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_and_cleanup_checkpoints(epoch_checkpoint, epoch_checkpoint_path, checkpoint_dir)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((480,640)),
        transforms.ToTensor(),
    ])
    yolo_model_lnn = DroneControlSystem(device=device)
    bert_model = TextProcessor(device=device)
    train_data_path = r"E:\Dataset\VisDrone2019-MOT-test-dev\sequences"

    image_path = "test1.jpg"  # Dumy Forward pass
    pil_image = Image.open(image_path).convert("RGB")
    tensor_image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        results = yolo_model_lnn.yolo(tensor_image)

    
    # Initialize models and training parameters
    train_dataset = ImageSequenceDataset(train_data_path, transform=transform, device=device, bert_model=bert_model)

    optimizer = optim.Adam(yolo_model_lnn.parameters(), lr=0.001)
    epochs = 10 #
    checkpoint_path = "./checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    yolo_model_lnn.new_sequence()
    train_model(yolo_model_lnn, train_dataset, epochs, optimizer, checkpoint_path, device, batch_size=20)