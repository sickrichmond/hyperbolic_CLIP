import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_1d(inputs, dim):
    inputs = inputs.movedim(dim, -1)
    length = inputs.shape[-1]
    reordered = torch.cat(
        (inputs[..., ::2], inputs[..., 1::2].flip(dims=(-1,))), dim=-1
    )
    spectrum = torch.fft.fft(reordered, dim=-1)
    frequencies = torch.arange(
        length, dtype=inputs.dtype, device=inputs.device
    )
    angles = torch.pi * frequencies / (2 * length)
    transformed = 2 * (
        spectrum.real * torch.cos(angles)
        + spectrum.imag * torch.sin(angles)
    )
    return transformed.movedim(-1, dim)


def _dct_2d(inputs):
    return _dct_1d(_dct_1d(inputs, -1), -2)


def get_input_data(images, input_data="dct"):
    if input_data == "img":
        return images
    if input_data == "dct":
        return _dct_2d(images)
    raise ValueError(f"Unsupported POSE input_data: {input_data}")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride == 2:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )

    def forward(self, images):
        return self.layers(images)


class POSEFeatureExtractor(nn.Module):
    """POSE's four-stage CNN, feature projection, and classification head."""

    def __init__(self, num_classes, input_data="dct", feature_dim=512):
        super().__init__()
        self.input_data = input_data
        self.features = nn.Sequential(
            _ConvBlock(3, 64, stride=2),
            _ConvBlock(64, 64),
            _ConvBlock(64, 128, stride=2),
            _ConvBlock(128, 128),
            _ConvBlock(128, 256, stride=2),
            _ConvBlock(256, 256),
            _ConvBlock(256, 512, stride=2),
            _ConvBlock(512, 512),
        )
        self.projection = nn.Linear(512 * 8 * 8, feature_dim)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_features(self, images):
        images = get_input_data(images, self.input_data)
        features = self.features(images)
        features = F.adaptive_avg_pool2d(features, (8, 8)).flatten(1)
        return self.projection(features)

    def classify(self, features):
        return self.classification_head(features)

    def forward(self, images):
        features = self.extract_features(images)
        return self.classify(features), features


class SingleLayerAugmenter(nn.Module):
    """Lightweight image-space augmentation network used by POSE."""

    def __init__(self, hidden_channels=32, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size, padding=padding),
            nn.Conv2d(hidden_channels, 3, kernel_size, padding=padding),
        )

    def forward(self, images):
        return self.layers(images)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        distances = torch.cdist(features, features, p=2)
        same_class = labels[:, None].eq(labels[None, :])
        positive_distances = distances.masked_fill(~same_class, float("-inf"))
        negative_distances = distances.masked_fill(same_class, float("inf"))
        hardest_positive = positive_distances.max(dim=1).values
        hardest_negative = negative_distances.min(dim=1).values
        return F.relu(hardest_positive - hardest_negative + self.margin).mean()
