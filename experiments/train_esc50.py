#!/usr/bin/env python3
"""
ESC-50 Audio Classification Training Script

This script trains a ResNet model on the ESC-50 dataset for audio classification.
The ESC-50 dataset contains 2000 environmental audio recordings across 50 classes.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from network_architectures import weak_mxh64_1024
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the 'src' directory is in the correct location")
    sys.exit(1)


class ESC50Dataset(Dataset):
    """ESC-50 Dataset class for loading audio files and labels"""
    
    def __init__(self, data_dir, metadata_file, fold, train=True, sample_rate=44100, duration=5.0):
        """
        Args:
            data_dir: Path to ESC-50 dataset directory
            metadata_file: Path to esc50.csv file
            fold: Fold number for cross-validation (1-5)
            train: If True, use all folds except 'fold' for training. If False, use only 'fold' for testing
            sample_rate: Target sample rate
            duration: Duration of audio clips in seconds
        """
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, 'audio')
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Filter data based on fold
        if train:
            self.data = self.metadata[self.metadata['fold'] != fold].reset_index(drop=True)
        else:
            self.data = self.metadata[self.metadata['fold'] == fold].reset_index(drop=True)
        
        print(f"{'Training' if train else 'Testing'} set: {len(self.data)} samples")
        
        # Create label mapping
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Load and process audio file"""
        row = self.data.iloc[idx]
        filename = row['filename']
        target = row['target']
        
        # Load audio file
        audio_path = os.path.join(self.audio_dir, filename)
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Pad or trim to target length
            if waveform.shape[1] > self.target_length:
                # Trim from center
                start = (waveform.shape[1] - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
            elif waveform.shape[1] < self.target_length:
                # Pad with zeros
                pad_length = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # Convert to mel-spectrogram manually to avoid issues with compute_melspec
            # Create mel-spectrogram transform
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                n_mels=128
            )
            amplitude_to_db = T.AmplitudeToDB()
            
            # Apply transforms
            melspec = mel_transform(waveform.squeeze(0))
            melspec = amplitude_to_db(melspec)
            melspec = melspec.T  # Transpose to match expected format
            
            return melspec, target
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros with correct mel-spectrogram shape
            melspec = torch.zeros((431, 128))  # Transposed shape: (time, freq)
            return melspec, target


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Add channel dimension if needed
        if len(data.shape) == 3:
            data = data.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        optimizer.zero_grad()
        output, _ = model(data)  # Model returns (output, intermediate_features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        predictions.extend(pred.cpu().numpy())
        targets.extend(target.cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # Add channel dimension if needed
            if len(data.shape) == 3:
                data = data.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
            output, _ = model(data)  # Model returns (output, intermediate_features)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    
    return avg_loss, accuracy, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Train ESC-50 Audio Classification Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ESC-50 dataset directory')
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Fold number for cross-validation (1-5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset paths
    metadata_file = os.path.join(args.data_dir, 'meta', 'esc50.csv')
    
    # Create datasets
    print(f"Creating datasets for fold {args.fold}...")
    train_dataset = ESC50Dataset(args.data_dir, metadata_file, args.fold, train=True)
    val_dataset = ESC50Dataset(args.data_dir, metadata_file, args.fold, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    # Create model
    print("Creating model...")
    # Define global pooling function (using functional average pooling)
    glplfn = nn.functional.avg_pool2d
    model = weak_mxh64_1024(nclass=50, glplfn=glplfn)  # ESC-50 has 50 classes
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, f'best_model_fold{args.fold}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'args': args
            }, model_path)
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f'training_history_fold{args.fold}.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 50)
    
    # Load best model for final evaluation
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get final predictions
    _, _, final_preds, final_targets = validate_epoch(model, val_loader, criterion, device)
    
    # Print classification report
    class_names = val_dataset.classes
    print("\nClassification Report:")
    print(classification_report(final_targets, final_preds, target_names=class_names))
    
    print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    main()