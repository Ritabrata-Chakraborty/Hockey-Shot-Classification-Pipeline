"""
TCN architecture for temporal sequence classification.
Self-contained implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..physics.shot_logical import TrajectoryPoint, get_all_shot_types


class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connections."""
    
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample:
            nn.init.kaiming_normal_(self.downsample.weight)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout1(out)
        out = self.relu(self.conv2(out))
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        # Trim output to match residual if needed
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                      stride=1, dilation=dilation_size,
                                      padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with dilated convolution for long-range dependencies."""
    
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=dilation*(kernel_size-1)//2, dilation=dilation)
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch, channels, time]
        residual = x
        
        # First conv
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [batch, time, channels] for LayerNorm
        out = self.ln1(out)
        out = out.transpose(1, 2)  # [batch, channels, time]
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)
        out = self.dropout(out)
        
        # Residual connection
        return self.relu(out + residual)


class HockeyTCNClassifier(nn.Module):
    """
    Improved Trajectory + Physics Fusion Network
    
    Enhancements for 4000 samples (1000 per class):
    - Dilated convolutions for long-range temporal context
    - Residual connections to prevent gradient vanishing
    - LayerNorm instead of BatchNorm for stability
    - Larger trajectory embedding (64) to balance with auxiliary (32)
    - Attention-based fusion instead of simple concatenation
    - Label smoothing in training (handled in trainer)
    """
    
    def __init__(self, temporal_input_size: int = 3, auxiliary_size: int = 52, 
                 num_classes: int = 4, dropout: float = 0.15):
        super().__init__()
        
        # Stage 1: Trajectory Encoder with Dilated Convolutions + Residuals
        # Progressive expansion with increasing receptive field
        self.stem = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Residual blocks with exponentially increasing dilation
        # Covers receptive field of ~200 timesteps
        self.res_blocks = nn.ModuleList([
            ResidualBlock(32, kernel_size=3, dilation=1, dropout=dropout),   # RF: 5
            ResidualBlock(32, kernel_size=3, dilation=2, dropout=dropout),   # RF: 13
            ResidualBlock(32, kernel_size=3, dilation=4, dropout=dropout),   # RF: 29
            ResidualBlock(32, kernel_size=3, dilation=8, dropout=dropout),   # RF: 61
            ResidualBlock(32, kernel_size=3, dilation=16, dropout=dropout),  # RF: 125
        ])
        
        # Progressive downsampling and expansion
        self.encoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Decoder phase
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Trajectory embedding projection (increased from 16 to 64)
        self.traj_projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stage 2: Auxiliary Feature Encoder (balanced with trajectory)
        self.aux_mlp = nn.Sequential(
            nn.Linear(auxiliary_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Stage 3: Attention-based Fusion
        # Learn to weight trajectory vs auxiliary features
        self.fusion_attention = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # 2 attention weights (traj, aux)
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.temporal_input_size = temporal_input_size
        self.auxiliary_size = auxiliary_size
        self.num_classes = num_classes
    
    def forward(self, temporal_x, auxiliary_x):
        """
        Args:
            temporal_x: [batch, 3, T] - x,y,z coordinates over time (T=200)
            auxiliary_x: [batch, 52] - aggregated physical features
        
        Returns:
            [batch, num_classes] - class logits
        """
        # Stage 1: Process temporal trajectory with dilated convolutions
        x = self.stem(temporal_x)  # [batch, 32, 200]
        
        # Apply residual blocks with increasing dilation
        for res_block in self.res_blocks:
            x = res_block(x)  # [batch, 32, 200]
        
        # Downsample and encode
        x = self.encoder(x)  # [batch, 64, T/4]
        
        # Global pooling (avg + max for robustness)
        traj_avg = x.mean(dim=-1)  # [batch, 64]
        traj_max = x.max(dim=-1)[0]  # [batch, 64]
        traj_feat = (traj_avg + traj_max) / 2  # [batch, 64]
        traj_feat = self.traj_projection(traj_feat)  # [batch, 64]
        
        # Stage 2: Process auxiliary features
        aux_feat = self.aux_mlp(auxiliary_x)  # [batch, 32]
        
        # Stage 3: Attention-based fusion
        fused = torch.cat([traj_feat, aux_feat], dim=1)  # [batch, 96]
        attention_weights = self.fusion_attention(fused)  # [batch, 2]
        
        # Apply attention (soft weighting)
        # Scale trajectory and auxiliary by learned weights
        traj_weighted = traj_feat * attention_weights[:, 0:1]
        aux_weighted = aux_feat * attention_weights[:, 1:2]
        
        # Final fusion
        fused_weighted = torch.cat([traj_weighted, aux_weighted], dim=1)  # [batch, 96]
        
        # Classification
        return self.classifier(fused_weighted)  # [batch, num_classes]


class TrajectoryDataset(Dataset):
    """Dataset for trajectory sequences."""
    
    def __init__(self, trajectories: List[List[TrajectoryPoint]],
                 labels: List[str], label_encoder: LabelEncoder,
                 scaler: StandardScaler = None, max_length: int = 200):
        self.trajectories = trajectories
        self.labels = labels
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.max_length = max_length
        
        self.sequences, self.encoded_labels = self._prepare_data()
    
    def _extract_temporal_features(self, trajectory: List[TrajectoryPoint]) -> np.ndarray:
        """
        Extract x,y,z coordinates over time with per-axis normalization
        and temporal alignment around apex (max height).
        """
        if len(trajectory) < 2:
            return np.zeros((3, self.max_length))
        
        # Extract coordinates
        coords = []
        for point in trajectory:
            coords.append([point.x, point.y, point.z])
        
        coords = np.array(coords)  # [timesteps, 3]
        
        # Find apex (max z) for temporal alignment
        apex_idx = np.argmax(coords[:, 2])
        
        # Center trajectory around apex for consistent temporal alignment
        # This helps TCN learn phase-aligned patterns
        # Split into ascent and descent phases
        ascent = coords[:apex_idx+1]
        descent = coords[apex_idx+1:]
        
        # Resample ascent and descent to fixed lengths (half each)
        half_length = self.max_length // 2
        
        if len(ascent) > half_length:
            indices_asc = np.linspace(0, len(ascent)-1, half_length, dtype=int)
            ascent = ascent[indices_asc]
        else:
            # Pad with first value
            padding = np.repeat([ascent[0]], half_length - len(ascent), axis=0)
            ascent = np.vstack([padding, ascent])
        
        if len(descent) > half_length:
            indices_desc = np.linspace(0, len(descent)-1, half_length, dtype=int)
            descent = descent[indices_desc]
        else:
            # Pad with last value
            padding = np.repeat([descent[-1] if len(descent) > 0 else ascent[-1]], 
                              half_length - len(descent), axis=0)
            descent = np.vstack([descent, padding])
        
        # Combine ascent + descent
        coords = np.vstack([ascent, descent])  # [max_length, 3]
        
        # Per-axis normalization (critical for TCN performance)
        # Each axis (x, y, z) normalized independently to N(0,1)
        for i in range(3):
            axis_mean = coords[:, i].mean()
            axis_std = coords[:, i].std() + 1e-8
            coords[:, i] = (coords[:, i] - axis_mean) / axis_std
        
        return coords.T  # [3, max_length] - transpose for Conv1d
    
    def _extract_auxiliary_features(self, trajectory: List[TrajectoryPoint]) -> np.ndarray:
        """Extract 52 aggregated features."""
        from .feature_extraction import extract_auxiliary_features
        return extract_auxiliary_features(trajectory)
    
    def _prepare_data(self):
        temporal_sequences = []
        auxiliary_features = []
        encoded_labels = []
        
        for traj, label in zip(self.trajectories, self.labels):
            # Extract temporal (x,y,z over time)
            temporal = self._extract_temporal_features(traj)
            temporal_sequences.append(temporal)
            
            # Extract auxiliary (52 aggregated features)
            auxiliary = self._extract_auxiliary_features(traj)
            auxiliary_features.append(auxiliary)
            
            # Encode label
            encoded_labels.append(self.label_encoder.transform([label])[0])
        
        temporal_sequences = np.array(temporal_sequences)  # [N, 3, 200]
        auxiliary_features = np.array(auxiliary_features)  # [N, 52]
        
        # Scale features
        if self.scaler is not None:
            # Scale temporal features
            n_samples, n_channels, n_timesteps = temporal_sequences.shape
            temporal_reshaped = temporal_sequences.reshape(n_samples, -1)  # [N, 600]
            temporal_scaled = self.scaler['temporal'].transform(temporal_reshaped)
            temporal_sequences = temporal_scaled.reshape(n_samples, n_channels, n_timesteps)
            
            # Scale auxiliary features
            auxiliary_features = self.scaler['auxiliary'].transform(auxiliary_features)
        
        return (temporal_sequences, auxiliary_features), np.array(encoded_labels)
    
    def __len__(self):
        temporal_seq, auxiliary_feat = self.sequences
        return len(temporal_seq)
    
    def __getitem__(self, idx):
        temporal_seq, auxiliary_feat = self.sequences
        temporal = torch.FloatTensor(temporal_seq[idx])  # [3, 200]
        auxiliary = torch.FloatTensor(auxiliary_feat[idx])  # [52]
        label = torch.LongTensor([self.encoded_labels[idx]])[0]
        return (temporal, auxiliary), label


@dataclass
class TCNTrainingConfig:
    """Configuration for TCN training - optimized for 4000 samples."""
    dropout: float = 0.15  # Moderate dropout for 4K samples
    learning_rate: float = 5e-4  # Lower LR for stable convergence
    batch_size: int = 32  # Larger batch for better gradient estimates
    num_epochs: int = 150  # More epochs for larger dataset
    patience: int = 15  # More patience for convergence
    device: str = 'auto'
    weight_decay: float = 1e-4  # Stronger regularization
    
    # Label smoothing to handle ambiguous classes
    label_smoothing: float = 0.1
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # LR scheduler params
    lr_patience: int = 10
    lr_factor: float = 0.5
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_tcn_model(config: TCNTrainingConfig, num_classes: int = None) -> HockeyTCNClassifier:
    """Create TCN model with given configuration."""
    if num_classes is None:
        num_classes = len(get_all_shot_types())
    
    temporal_input_size = 3  # x, y, z coordinates
    auxiliary_size = 52  # Aggregated features
    
    return HockeyTCNClassifier(
        temporal_input_size=temporal_input_size,
        auxiliary_size=auxiliary_size,
        num_classes=num_classes,
        dropout=config.dropout
    )


__all__ = [
    'HockeyTCNClassifier',
    'TemporalConvNet',
    'TrajectoryDataset',
    'TCNTrainingConfig',
    'create_tcn_model'
]
