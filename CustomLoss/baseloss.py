
import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self,classification_criterion,reduction='mean'):
        super(BaseLoss,self).__init__()
        self.reduction=reduction
        self.classification_criterion=classification_criterion.__class__(reduction=reduction)
