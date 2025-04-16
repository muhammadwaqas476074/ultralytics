import torch.nn as nn
import torch
import os


class SpatialAwareMLP(nn.Module):
    def __init__(self, input_dim=None, reduced_dim=None, spatial_height=None, spatial_width=None, model_path=None):
        """
        Initialize SpatialAwareMLP model
        
        Args:
            input_dim: Input dimension size (required if model_path is None)
            reduced_dim: Reduced output dimension size (required if model_path is None)
            spatial_height: Original spatial height (required if model_path is None)
            spatial_width: Original spatial width (required if model_path is None)
            model_path: Path to load a saved model (if provided, other params are optional)
        """
        super().__init__()
        
        # Check if loading from a path
        if model_path is not None:
            self._load_from_path(model_path)
        else:
            # Validate required parameters for creating a new model
            if None in (input_dim, reduced_dim, spatial_height, spatial_width):
                raise ValueError("When model_path is not provided, all of input_dim, reduced_dim, spatial_height, and spatial_width must be specified")
            
            # Create a new model
            self._create_new_model(input_dim, reduced_dim, spatial_height, spatial_width)

    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
        """
        # Reshape spatial dimensions to feature dimension
        batch_size, channels, height, width = x.shape
        x_flattened = x.reshape(batch_size, channels, -1).permute(0, 2, 1)  # [batch, h*w, channels]
        x_flattened = x_flattened.reshape(-1, channels)  # [batch*h*w, channels]
        
        # Pass through MLP
        reduced_features = self.spatial_mlp(x_flattened)  # [batch*h*w, reduced_dim]
        
        # Reshape back to spatial dimensions
        reduced_features = reduced_features.reshape(batch_size, self.spatial_height * self.spatial_width, -1)
        reduced_features = reduced_features.permute(0, 2, 1).reshape(batch_size, self.reduced_dim, height, width)
        # Apply spatial encoding
        enhanced_features = reduced_features + self.spatial_encoding
        
        return enhanced_features
    
    def _create_new_model(self, input_dim, reduced_dim, spatial_height, spatial_width):
        """Initialize a new model with given parameters"""
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        
        self.spatial_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            
            nn.Linear(input_dim // 2, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU()
        )
        
        self.spatial_encoding = nn.Parameter(
            torch.randn(1, reduced_dim, spatial_height, spatial_width)
        )
    
    def _load_from_path(self, model_path):
        """Load model from a saved checkpoint"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        config = checkpoint['model_config']
        self.input_dim = config['input_dim']
        self.reduced_dim = config['reduced_dim']
        self.spatial_height = config['spatial_height']
        self.spatial_width = config['spatial_width']
        
        # Create model architecture first
        self._create_new_model(
            self.input_dim,
            self.reduced_dim,
            self.spatial_height, 
            self.spatial_width
        )
    
        # Then load state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Store training info if available
        self.last_epoch = checkpoint.get('epoch', None)
        self.last_loss = checkpoint.get('loss', None)
        
        print(f"Model loaded from {model_path}")
    
    def save(self, filepath, optimizer=None, epoch=None, loss=None):
        """
        Save the model to disk
        
        Args:
            filepath: Path where to save the model
            optimizer: Optional optimizer to save its state
            epoch: Optional current epoch number
            loss: Optional current loss value
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare state dictionary
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'reduced_dim': self.reduced_dim,
                'spatial_height': self.spatial_height,
                'spatial_width': self.spatial_width
            }
        }
        
        # Add optimizer and training info if provided
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        
        # Save the checkpoint
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
