"""
MultiCue-DF Model Architectures.

Model 1 — MLPBaseline         : flatten stream1 → 3 FC layers
Model 2 — SingleCNNBaseline   : 4-block CNN on stream1
Model 3 — Stream1CNN          : ResNet-18 feature extractor (512-d)
Model 4 — Stream2CNN          : Custom 4-block CNN (256-d)
Model 5 — Stream3CNN          : Shallow 3-layer CNN with LeakyReLU (128-d)
Model 6 — ChannelAttention    : SE-style attention on fused vector
Model 7 — MultiCueDFModel     : Full fusion model with ablation flags
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ── Initialisation helpers ────────────────────────────────────────────────────

def _he_relu(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _he_leaky(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                nonlinearity="leaky_relu", a=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ── Model 1 — MLP Baseline ────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """Flatten stream1 (224×224×3 = 150,528) through 3 FC layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150_528, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self.net.apply(_he_relu)

    def forward(self, x1, x2=None, x3=None):
        return self.net(x1)


# ── Model 2 — Single CNN Baseline ─────────────────────────────────────────────

class SingleCNNBaseline(nn.Module):
    """4-block CNN on stream1, outputs class logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
        self.apply(_he_relu)

    def forward(self, x1, x2=None, x3=None):
        return self.classifier(self.features(x1))


# ── Model 3 — Stream1CNN (ResNet-18 backbone) ─────────────────────────────────

class Stream1CNN(nn.Module):
    """
    ResNet-18 with pretrained ImageNet weights.
    Final FC replaced with FC(512→512) → ReLU → Dropout(0.3).
    First 6 children frozen (conv1, bn1, relu, maxpool, layer1, layer2).
    Output: 512-d feature vector.
    """

    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace classifier head with feature projector
        base.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        nn.init.kaiming_normal_(base.fc[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(base.fc[0].bias)

        # Freeze: conv1, bn1, relu, maxpool, layer1, layer2 (children 0-5)
        for child in list(base.children())[:6]:
            for p in child.parameters():
                p.requires_grad = False

        self.model = base

    def forward(self, x):
        return self.model(x)  # (B, 512)


# ── Model 4 — Stream2CNN (Eye+Mouth, Custom 4-block) ─────────────────────────

def _conv_block(in_ch, out_ch, pool="max"):
    """Two conv→BN→ReLU layers followed by pooling."""
    layers = [
        nn.Conv2d(in_ch,  out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    ]
    if pool == "max":
        layers.append(nn.MaxPool2d(2))
    else:
        layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)


class Stream2CNN(nn.Module):
    """
    Custom 4-block CNN for eye+mouth region.
    Output: 256-d feature vector.
    """

    def __init__(self):
        super().__init__()
        self.block1 = _conv_block(3,   32,  pool="max")
        self.block2 = _conv_block(32,  64,  pool="max")
        self.block3 = _conv_block(64,  128, pool="max")
        self.block4 = _conv_block(128, 256, pool="avg")
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.apply(_he_relu)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)  # (B, 256)


# ── Model 5 — Stream3CNN (FFT Frequency Domain) ───────────────────────────────

class Stream3CNN(nn.Module):
    """
    Shallow 3-layer CNN with LeakyReLU for FFT magnitude images.
    Output: 128-d feature vector.
    """

    def __init__(self):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32,  3, padding=1), nn.BatchNorm2d(32),  lrelu(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  lrelu(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), lrelu(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),
        )
        self.apply(_he_leaky)

    def forward(self, x):
        return self.head(self.features(x))  # (B, 128)


# ── Model 6 — Channel Attention (SE-style) ────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation attention over the fused feature vector.
    Learns per-stream contribution weights.
    """

    def __init__(self, in_features: int = 896, reduction: int = 16):
        super().__init__()
        mid = max(1, in_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_features, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_features),
            nn.Sigmoid(),
        )
        self.fc.apply(_he_relu)

    def forward(self, x):
        return self.fc(x)   # attention weights, same shape as x


# ── Model 7 — MultiCueDFModel (Full Fusion) ───────────────────────────────────

class MultiCueDFModel(nn.Module):
    """
    Fusion of up to 3 stream CNNs.
    Ablation flags (use_stream1/2/3) allow disabling individual streams.
    Attention is applied only when all 3 streams are active (fusion_dim==896).
    """

    _STREAM_DIMS = {1: 512, 2: 256, 3: 128}

    def __init__(
        self,
        dropout: float = 0.5,
        use_attention: bool = True,
        use_stream1: bool = True,
        use_stream2: bool = True,
        use_stream3: bool = True,
    ):
        super().__init__()
        assert use_stream1 or use_stream2 or use_stream3, \
            "At least one stream must be active."

        self.use_stream1   = use_stream1
        self.use_stream2   = use_stream2
        self.use_stream3   = use_stream3

        if use_stream1:
            self.stream1 = Stream1CNN()
        if use_stream2:
            self.stream2 = Stream2CNN()
        if use_stream3:
            self.stream3 = Stream3CNN()

        active_dims = (
            ([512] if use_stream1 else []) +
            ([256] if use_stream2 else []) +
            ([128] if use_stream3 else [])
        )
        self.fusion_dim = sum(active_dims)

        # Attention only makes sense when all 3 streams are fused (896-d)
        self._apply_attention = use_attention and (self.fusion_dim == 896)
        if self._apply_attention:
            self.attention = ChannelAttention(896, reduction=16)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.classifier.apply(_he_relu)

    def forward(self, x1, x2, x3):
        feats = []
        if self.use_stream1:
            feats.append(self.stream1(x1))
        if self.use_stream2:
            feats.append(self.stream2(x2))
        if self.use_stream3:
            feats.append(self.stream3(x3))

        fused = torch.cat(feats, dim=1)

        if self._apply_attention:
            fused = fused * self.attention(fused)

        return self.classifier(fused)
