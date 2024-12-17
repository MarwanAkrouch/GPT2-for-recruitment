import os
from datetime import datetime
import numpy as np
import torch

class CheckpointManager:
    def __init__(self, save_dir='checkpoints', model_config=None, additional_info=None):
        self.base_dir = save_dir
        self.model_config = model_config
        self.model_id = self._get_model_identifier(additional_info)

        # Create model-specific directory
        self.save_dir = os.path.join(self.base_dir, self.model_id)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _get_model_identifier(self, additional_info):
        """Create identifier from model type and key hyperparameters"""
        if self.model_config is None:
            return "default_model"
            
        # Extract key parameters
        model_type = self.model_config.model_type
        block_size = self.model_config.block_size
        n_layer = self.model_config.n_layer
        n_head = self.model_config.n_head
        n_embd = self.model_config.n_embd  # Added this line
        
        # Create identifier
        identifier = f"{model_type}_b{block_size}_l{n_layer}_h{n_head}_e{n_embd}_{additional_info}"  # Modified this line
        return identifier
        
    def save_checkpoint(self, model, optimizer, epoch, iter_num, best_val_metric, metrics, is_best=False):
        """Save model checkpoint with informative name"""
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': self.model_config,
            'epoch': epoch,
            'iter_num': iter_num,
            'best_val_metric': best_val_metric,
            'metrics': metrics
        }
        
        # Save latest checkpoint in model directory
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None, map_location=None):
        """Load model checkpoint with optional optimizer loading"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, f'latest.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            original_state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            
            # quick fix for loading model state dict problem
            for k, v in original_state_dict.items():
                # Remove 'transformer.base_model.model.' prefix
                if k.startswith('transformer.base_model.model.'):
                    new_key = k.replace('transformer.base_model.model.', 'transformer.')
                elif k.startswith('transformer.base_model.'):
                    new_key = k.replace('transformer.base_model.', 'transformer.')
                else:
                    new_key = k
                
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
                
        # Load optimizer state if optimizer is provided and state exists
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
        return checkpoint

    def list_checkpoints(self):
        """List all available checkpoints for this model"""
        if not os.path.exists(self.save_dir):
            print(f"No checkpoints found for model {self.model_id}")
            return
            
        print(f"\nCheckpoints for model {self.model_id}:")
        for file in os.listdir(self.save_dir):
            if file.endswith('.pt'):
                path = os.path.join(self.save_dir, file)
                size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                print(f"- {file} ({size:.1f} MB)")

def load_model(model_dir, checkpoint_name, device = 'cpu'):
    from model import GPTMatchingModel

    device = torch.device(device)
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    
    # Load checkpoint with explicit CPU mapping
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model_config = checkpoint['model_config']

    model_config.model_type = None
    model = GPTMatchingModel(model_config)

    checkpoint_manager = CheckpointManager(model_dir, model_config=model_config)
    
    # Modify the load_checkpoint method to include map_location
    checkpoint = checkpoint_manager.load_checkpoint(
        model, 
        checkpoint_path=checkpoint_path,
        map_location=torch.device(device)
    )
    
    print(f"Using device: {device}")
    return model, device
