"""
ResNet-50 based model for Person Re-Identification.
Supports both training with classification head and embedding extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Optional

import config


class TripletLoss(nn.Module):
    """
    Triplet loss with hard mining.
    """

    def __init__(self, margin: float = 0.3):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with hard mining.

        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Person IDs (batch_size,)

        Returns:
            Triplet loss value
        """
        # Compute pairwise distance matrix
        dist_mat = self._pairwise_distance(embeddings)

        # For each anchor, find the hardest positive and negative
        loss = self._hard_mining(dist_mat, labels)

        return loss

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distance.

        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)

        Returns:
            Distance matrix (batch_size, batch_size)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute dot product
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Compute squared distance
        squared_norm = torch.diag(dot_product)
        dist_mat = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
        dist_mat = torch.clamp(dist_mat, min=0.0)

        # Fix numerical errors
        mask = torch.eq(dist_mat, 0.0).float()
        dist_mat = dist_mat + mask * 1e-16

        dist_mat = torch.sqrt(dist_mat)

        # Correct the gradients
        dist_mat = dist_mat * (1.0 - mask)

        return dist_mat

    def _hard_mining(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Hard mining for triplet loss.

        Args:
            dist_mat: Distance matrix (batch_size, batch_size)
            labels: Person IDs (batch_size,)

        Returns:
            Triplet loss value
        """
        batch_size = dist_mat.size(0)

        # For each anchor, find hardest positive and negative
        dist_ap = []
        dist_an = []

        for i in range(batch_size):
            # Get distances for anchor i
            dist_i = dist_mat[i]

            # Positive mask (same identity)
            pos_mask = labels == labels[i]
            pos_mask[i] = False  # Exclude the anchor itself

            # Negative mask (different identity)
            neg_mask = labels != labels[i]

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Hardest positive (farthest positive)
                hardest_pos_dist = dist_i[pos_mask].max()
                dist_ap.append(hardest_pos_dist)

                # Hardest negative (closest negative)
                hardest_neg_dist = dist_i[neg_mask].min()
                dist_an.append(hardest_neg_dist)

        if len(dist_ap) == 0:
            # No valid triplets in this batch
            return torch.tensor(0.0, device=dist_mat.device, requires_grad=True)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


class ReIDModel(nn.Module):
    """
    ResNet-50 based Person Re-Identification model.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout_rate: float = 0.5):
        """
        Initialize Re-ID model.

        Args:
            num_classes: Number of person identities
            embedding_dim: Dimension of the embedding vector
            dropout_rate: Dropout rate
        """
        super(ReIDModel, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Embedding layers
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (batch_size, 3, height, width)
            return_embeddings: If True, return embeddings only

        Returns:
            If return_embeddings is True: embeddings (batch_size, embedding_dim)
            Otherwise: (logits, embeddings)
        """
        # Extract features
        features = self.backbone(x)  # (batch_size, 2048, H, W)

        # Global average pooling
        features = self.gap(features)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)

        # Get embeddings
        embeddings = self.bottleneck(features)  # (batch_size, embedding_dim)

        if return_embeddings:
            return embeddings

        # Classification
        logits = self.classifier(embeddings)  # (batch_size, num_classes)

        return logits, embeddings


def create_model(num_classes: int, device: torch.device) -> ReIDModel:
    """
    Create and initialize a Re-ID model.

    Args:
        num_classes: Number of person identities
        device: Device to place the model on

    Returns:
        Initialized ReIDModel
    """
    model = ReIDModel(
        num_classes=num_classes,
        embedding_dim=config.MODEL_CONFIG["embedding_dim"],
        dropout_rate=config.MODEL_CONFIG["dropout_rate"]
    )

    model = model.to(device)

    return model


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross-Entropy + Triplet Loss.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        triplet_weight: float = 1.0,
        triplet_margin: float = 0.3
    ):
        """
        Initialize combined loss.

        Args:
            num_classes: Number of person identities
            ce_weight: Weight for cross-entropy loss
            triplet_weight: Weight for triplet loss
            triplet_margin: Margin for triplet loss
        """
        super(CombinedLoss, self).__init__()

        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Classification logits (batch_size, num_classes)
            embeddings: Feature embeddings (batch_size, embedding_dim)
            labels: Person IDs (batch_size,)

        Returns:
            Tuple of (total_loss, ce_loss, triplet_loss)
        """
        ce = self.ce_loss(logits, labels)
        triplet = self.triplet_loss(embeddings, labels)

        total = self.ce_weight * ce + self.triplet_weight * triplet

        return total, ce, triplet


if __name__ == "__main__":
    """Test the model."""
    print("Testing Re-ID model...")

    # Create a dummy model
    num_classes = 751
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = create_model(num_classes, device)
    print(f"✓ Model created with {num_classes} classes")

    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 256, 128).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    print(f"✓ Input shape: {dummy_input.shape}")

    # Training mode
    model.train()
    logits, embeddings = model(dummy_input)
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        embeddings = model(dummy_input, return_embeddings=True)
    print(f"✓ Inference embeddings shape: {embeddings.shape}")

    # Test loss
    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=config.LOSS_CONFIG["ce_weight"],
        triplet_weight=config.LOSS_CONFIG["triplet_weight"],
        triplet_margin=config.LOSS_CONFIG["triplet_margin"]
    )
    criterion = criterion.to(device)

    model.train()
    logits, embeddings = model(dummy_input)
    total_loss, ce_loss, triplet_loss = criterion(logits, embeddings, dummy_labels)

    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ CE loss: {ce_loss.item():.4f}")
    print(f"✓ Triplet loss: {triplet_loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    print("\n✓ All tests passed!")
