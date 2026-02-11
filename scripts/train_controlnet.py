"""
ControlNet Training Script for Steel Defect Augmentation

This script implements the training pipeline from PROJECT(control_net).md.
It trains a ControlNet model to generate realistic steel defects conditioned on:
- Multi-channel hint images (defect shape + background structure + texture)
- Text prompts describing defect-background combinations

The trained model will generate physically plausible defects that respect
the underlying surface patterns and textures.

Usage:
    python scripts/train_controlnet.py --data_dir data/processed/controlnet_dataset
    python scripts/train_controlnet.py --data_dir data/processed/controlnet_dataset --resume checkpoints/last.ckpt
"""
import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image


class SteelDefectControlNetDataset(Dataset):
    """
    Dataset for ControlNet training with steel defects.
    
    Loads:
    - Source images (ROI patches)
    - Target images (same as source for this task)
    - Hint images (3-channel conditioning)
    - Text prompts
    """
    
    def __init__(self, jsonl_path: Path, base_dir: Path, 
                 image_size: int = 512, augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            jsonl_path: Path to train.jsonl
            base_dir: Base directory for relative paths
            image_size: Image size for training
            augment: Whether to apply data augmentation
        """
        self.base_dir = base_dir
        self.image_size = image_size
        self.augment = augment
        
        # Load training data
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.hint_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load source image
        source_path = self.base_dir / sample['source']
        source_image = Image.open(source_path).convert('RGB')
        
        # Load hint image
        hint_path = self.base_dir / sample['hint']
        hint_image = Image.open(hint_path).convert('RGB')
        
        # Apply transforms
        source_tensor = self.image_transform(source_image)
        target_tensor = source_tensor.clone()  # Target = source for defect generation
        hint_tensor = self.hint_transform(hint_image)
        
        # Get text prompt
        prompt = sample['prompt']
        
        return {
            'source': source_tensor,
            'target': target_tensor,
            'hint': hint_tensor,
            'prompt': prompt,
            'image_id': str(source_path.stem)
        }


class SimpleControlNet(nn.Module):
    """
    Simplified ControlNet architecture for demonstration.
    
    In production, you would use the official ControlNet implementation
    from https://github.com/lllyasviel/ControlNet
    
    This is a placeholder architecture that shows the key components:
    - Encoder: Processes hint image
    - Control: Generates control signals
    - Decoder: Generates output image conditioned on hints
    """
    
    def __init__(self, in_channels=3, out_channels=3, hint_channels=3):
        super(SimpleControlNet, self).__init__()
        
        # Hint encoder
        self.hint_encoder = nn.Sequential(
            nn.Conv2d(hint_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Main encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Control injection (zero-initialized for gradual training)
        self.control_conv = nn.Conv2d(256, 256, kernel_size=1)
        nn.init.zeros_(self.control_conv.weight)
        nn.init.zeros_(self.control_conv.bias)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, hint):
        """
        Forward pass.
        
        Args:
            x: Source image (B, 3, H, W)
            hint: Hint image (B, 3, H, W)
            
        Returns:
            Generated image (B, 3, H, W)
        """
        # Encode hint
        hint_features = self.hint_encoder(hint)
        
        # Encode source
        x_features = self.encoder(x)
        
        # Apply control
        control = self.control_conv(hint_features)
        controlled_features = x_features + control
        
        # Decode
        output = self.decoder(controlled_features)
        
        return output


class ControlNetTrainer:
    """
    Trainer for ControlNet model.
    """
    
    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-4, device='cuda', output_dir='outputs'):
        """
        Initialize trainer.
        
        Args:
            model: ControlNet model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            lr: Learning rate
            device: Device for training
            output_dir: Output directory for checkpoints and logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            hint = batch['hint'].to(self.device)
            
            # Forward pass
            output = self.model(source, hint)
            
            # Compute loss
            loss = self.l1_loss(output, target) + 0.5 * self.l2_loss(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            hint = batch['hint'].to(self.device)
            
            output = self.model(source, hint)
            loss = self.l1_loss(output, target) + 0.5 * self.l2_loss(output, target)
            
            val_loss += loss.item()
        
        avg_loss = val_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, num_epochs, save_every=5):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print("="*80)
        print("Starting ControlNet Training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Total epochs: {num_epochs}")
        print("="*80)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best.pth')
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save latest
            self.save_checkpoint('last.pth')
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Train ControlNet for steel defect augmentation'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing ControlNet dataset (with train.jsonl)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Image size for training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/controlnet_training',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training (cuda or cpu)'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / 'train.jsonl'
    
    if not jsonl_path.exists():
        print(f"Error: train.jsonl not found at {jsonl_path}")
        print("Please run scripts/prepare_controlnet_data.py first")
        return
    
    print("="*80)
    print("ControlNet Training Setup")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"JSONL path: {jsonl_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = SteelDefectControlNetDataset(
        jsonl_path=jsonl_path,
        base_dir=data_dir.parent,  # Go up one level from controlnet_dataset
        image_size=args.image_size,
        augment=True
    )
    
    print(f"Loaded {len(dataset)} training samples")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create model
    print("\nInitializing model...")
    model = SimpleControlNet(
        in_channels=3,
        out_channels=3,
        hint_channels=3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ControlNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_every=args.save_every)
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Evaluate the trained model")
    print("2. Generate synthetic defects using the trained ControlNet")
    print("3. Use generated data to augment the training set")


if __name__ == '__main__':
    main()
