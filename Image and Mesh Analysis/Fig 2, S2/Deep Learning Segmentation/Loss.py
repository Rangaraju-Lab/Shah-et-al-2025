import torch
import torch.nn as nn
import torch.nn.functional as F

class RebalancedMaskedBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with class-rebalancing and masking support, compatible with mixed precision.
    """
    def __init__(self, pos_weight=None, size_average=True, max_weight=5.0):
        super().__init__()
        self.size_average = size_average
        self.pos_weight = pos_weight  # Optional: Predefined positive class weight
        self.max_weight = max_weight  # Limit on dynamic weight

    def forward(self, input, target, mask=None):
        # Compute positive and negative class counts dynamically
        positive_count = (target == 1).sum().float()
        negative_count = (target == 0).sum().float()

        # Calculate the positive weight (clamped to prevent extreme values)
        if positive_count > 0:
            pos_weight = torch.clamp(negative_count / positive_count, max=self.max_weight)
        else:
            pos_weight = 1.0  # Handle all-background patches gracefully

        # Use predefined weight if provided
        if self.pos_weight is not None:
            pos_weight = self.pos_weight

        # Apply class weighting
        weight = torch.where(target == 1, pos_weight, 1.0).to(input.device)

        # Compute raw BCE loss with logits and weighting
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='none')

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask  # Zero out loss for padded areas
            if self.size_average:
                return loss.sum() / mask.sum()  # Average over valid areas
            else:
                return loss.sum()

        # Standard reduction if no mask is applied
        return loss.mean() if self.size_average else loss.sum()
