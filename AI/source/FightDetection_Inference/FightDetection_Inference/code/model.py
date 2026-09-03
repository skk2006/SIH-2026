
import torch
import torch.nn as nn

INPUT_SIZE = 175
WINDOW_SIZE = 30

NUM_KEYPOINTS = 17
POSE_SIZE = NUM_KEYPOINTS * 2

CLASS_NAMES = ["NonFight", "Fight"]


class PoseTemporalClassifier(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        dropout=0.30
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last = output[:, -1, :]

        logits = self.classifier(last)

        return logits.squeeze(1)
