import torch
import torch.nn as nn 
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    The modulating factor (1 - p_t)^y reduces the loss contribution from easy examples and extends the range in which an example receives low loss.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where:
            - p_t is the model's estimated probability for the true class
            - alpha_t is the class weight for the true class
            - gamma >= 0 is the focusing parameter

    Citation:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE 
            International Conference on Computer Vision (pp. 2980-2988). https://arxiv.org/abs/1708.02002
    '''
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute Focal Loss

        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(
            input=inputs,
            target=targets,
            weight=self.alpha,
            reduction='none'
        )

        # Calculate probabilities
        # p_t is probability of the true class 
        p_t = torch.exp(-ce_loss)


        # Calculate Focal Loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
def create_focal_loss(gamma=2.0, device='cuda'):
    # Class weights: [Safe, Ambiguous, Unsafe]
    # Moderate weights since Focal Loss provides additional focusing via gamma
    # Ambiguous class gets 5-7x higher weight to compensate for rarity
    
    weights_overall = torch.tensor([1.5, 5.0, 1.0], dtype=torch.float).to(device)
    weights_harmful = torch.tensor([1.8, 6.0, 1.0], dtype=torch.float).to(device)
    weights_bias = torch.tensor([2.0, 7.0, 1.0], dtype=torch.float).to(device)
    weights_policy = torch.tensor([2.0, 7.0, 1.0], dtype=torch.float).to(device)
    
    weighted_losses = {
        'Q_overall': FocalLoss(alpha=weights_overall, gamma=gamma),
        'Q2_harmful': FocalLoss(alpha=weights_harmful, gamma=gamma),
        'Q3_bias': FocalLoss(alpha=weights_bias, gamma=gamma),
        'Q6_policy': FocalLoss(alpha=weights_policy, gamma=gamma)
    }
    
    return weighted_losses