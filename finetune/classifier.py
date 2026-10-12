"""Classifier head on top of the SSL-pretrained backbone, plus a loader
that reconstructs the backbone architecture from a saved SSL checkpoint."""

import torch
import torch.nn as nn

from ssl_pretrain.simclr_model import build_backbone


class RemoteSensingClassifier(nn.Module):
    def __init__(self, arch: str = "resnet18", num_classes: int = 10, small_image: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch, small_image=small_image)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def load_pretrained_backbone(model: RemoteSensingClassifier, ssl_checkpoint_path: str, device):
    """Loads SSL-pretrained backbone weights (from train_ssl.py) into a
    RemoteSensingClassifier, ignoring the projection head."""
    ckpt = torch.load(ssl_checkpoint_path, map_location=device)
    ssl_state_dict = ckpt["model_state_dict"]

    backbone_state_dict = {
        k.replace("backbone.", ""): v
        for k, v in ssl_state_dict.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = model.backbone.load_state_dict(backbone_state_dict, strict=False)
    if missing or unexpected:
        print(f"[load_pretrained_backbone] missing={missing} unexpected={unexpected}")
    return model
