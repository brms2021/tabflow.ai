"""Model architectures for guitar tablature training.

1. GuitarPitchNet: CRNN for guitar-specific pitch detection
2. FretNet: Transformer for MIDI -> (string, fret) assignment
3. TechniqueNet: CNN for playing technique classification
"""

import torch
import torch.nn as nn

from .dataset import NUM_PITCHES


class GuitarPitchNet(nn.Module):
    """CRNN for guitar pitch detection.

    Architecture: CQT input -> CNN encoder -> BiLSTM -> per-frame pitch sigmoid.
    Similar to Onsets and Frames but tuned for guitar frequency range.

    Input: (batch, n_bins, context_frames) CQT magnitude
    Output: (batch, NUM_PITCHES) sigmoid activation per pitch
    """

    def __init__(
        self,
        n_bins: int = 264,
        context_frames: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, context_frames)),
        )

        self.lstm = nn.LSTM(
            input_size=128 * context_frames,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, NUM_PITCHES),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bins, context_frames)
        x = x.unsqueeze(1)  # (batch, 1, n_bins, context_frames)
        x = self.cnn(x)  # (batch, 128, 1, context_frames)
        x = x.squeeze(2)  # (batch, 128, context_frames)
        x = x.view(x.size(0), 1, -1)  # (batch, 1, 128*context_frames)
        x, _ = self.lstm(x)  # (batch, 1, hidden*2)
        x = x.squeeze(1)  # (batch, hidden*2)
        return self.fc(x)  # (batch, NUM_PITCHES)


class FretNet(nn.Module):
    """Transformer for MIDI note sequence -> (string, fret) assignment.

    Architecture: note features -> embedding -> Transformer encoder -> dual classification heads.

    Input: (batch, seq_len, 3) — pitch, onset_delta, duration
    Output: (batch, seq_len, 6) string logits + (batch, seq_len, 25) fret logits
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 64,
        num_strings: int = 6,
        max_fret: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.string_head = nn.Linear(d_model, num_strings)
        self.fret_head = nn.Linear(d_model, max_fret + 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, 3)
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)  # (batch, seq_len, d_model)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions)

        x = self.transformer(x)  # (batch, seq_len, d_model)

        string_logits = self.string_head(x)  # (batch, seq_len, 6)
        fret_logits = self.fret_head(x)  # (batch, seq_len, 25)

        return string_logits, fret_logits


class TechniqueNet(nn.Module):
    """CNN for guitar technique classification.

    Architecture: mel-spectrogram -> 1D CNN -> global avg pool -> FC -> softmax.

    Input: (batch, n_mels, n_frames) mel spectrogram
    Output: (batch, num_classes) logits
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 12,  # matches TechniqueDataset.TECHNIQUE_CLASSES
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, n_frames)
        x = self.features(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        return self.classifier(x)  # (batch, num_classes)
