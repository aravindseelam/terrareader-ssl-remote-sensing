"""
SimCLR model definition: a ResNet backbone adapted for small (64x64) satellite
tiles, plus a projection head and the NT-Xent contrastive loss.

Standard torchvision ResNets use a 7x7 stride-2 conv + maxpool stem, which is
tuned for 224x224 ImageNet images. On 64x64 tiles that stem throws away too
much spatial resolution before the first residual block ever sees the image,
so we replace it with a 3x3 stride-1 conv and drop the maxpool -- a standard
trick (used in the original SimCLR paper's CIFAR experiments) for small-image
contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


def build_backbone(arch: str = "resnet18", small_image: bool = True):
    if arch == "resnet18":
        model = resnet18(weights=None)
        feat_dim = 512
    elif arch == "resnet50":
        model = resnet50(weights=None)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {arch}")

    if small_image:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Identity()  # remove classification head; we use it as a feature extractor
    return model, feat_dim


class ProjectionHead(nn.Module):
    """Standard 2-layer MLP projection head used by SimCLR."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, arch: str = "resnet18", proj_dim: int = 128, small_image: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch, small_image=small_image)
        self.projection_head = ProjectionHead(feat_dim, hidden_dim=feat_dim, out_dim=proj_dim)
        self.feat_dim = feat_dim

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss (NT-Xent), the SimCLR
    contrastive loss. z1 and z2 are projections of the two augmented views,
    shape (batch_size, proj_dim) each.
    """
    batch_size = z1.size(0)
    device = z1.device

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)  # (2N, D)
    similarity_matrix = torch.matmul(representations, representations.T)  # (2N, 2N)
    similarity_matrix = similarity_matrix / temperature

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity_matrix.masked_fill_(mask, float("-inf"))

    # Positive pair for row i in [0, N) is at index i+N, and vice versa
    positive_indices = torch.arange(2 * batch_size, device=device)
    positive_indices = (positive_indices + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(similarity_matrix, positive_indices)
    return loss
