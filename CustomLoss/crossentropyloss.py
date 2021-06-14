import torch
from CustomLoss.baseloss import BaseLoss
import torch.nn.functional as F
import torch.nn as nn
class CrossEntropyLoss(nn.Module):
    def __init__(self, custom_loss_reduction):
        super(CrossEntropyLoss,self).__init__()
        self.custom_loss_reduction=custom_loss_reduction

    def forward(self,prediction,target):
        crossentropy=torch.diagonal(torch.matmul(-F.log_softmax(prediction,dim=1),target.T))
        if self.custom_loss_reduction=='mean':
            crossentropy=crossentropy.mean()
        elif self.custom_loss_reduction=='sum':
            crossentropy=crossentropy.sum()

        return crossentropy

class ModifyTargetCrossEntropyLoss(BaseLoss):
    def __init__(self, classification_criterion,custom_loss_reduction, reduction):
        super(ModifyTargetCrossEntropyLoss,self).__init__(classification_criterion, reduction=reduction)
        self.custom_loss_reduction=custom_loss_reduction
        self.modify_target_criterion=CrossEntropyLoss(custom_loss_reduction)

    def forward(self,prediction,target):
        #basic
        classification_loss=self.classification_criterion(prediction,target)
        #custom
        modifytarget=torch.zeros_like(prediction)
        target_index=torch.ones_like(target).unsqueeze(-1).cumsum(dim=0)-1
        target=target.unsqueeze(-1)
        
        modifytarget[target_index,target]=1#참

        modifytarget_loss=self.modify_target_criterion(prediction,modifytarget)
        loss=classification_loss+modifytarget_loss
        return loss


def covariance_loss(logits, labels, T, device):
    bsz, n_cats, n_heads = logits.size()
    if n_heads < 2:
        return 0
    all_probs = torch.softmax(logits/T, dim=1)
    label_inds = torch.ones(bsz, n_cats).cuda(device)
    label_inds[range(bsz), labels] = 0
    # removing the ground truth prob
    probs = all_probs * label_inds.unsqueeze(-1).detach()
    # re-normalize such that probs sum to 1
    #probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = (torch.softmax(logits/T, dim=1) + 1e-8)
    # cosine regularization
    #### I added under 2-line
    probs -= probs.mean(dim=1, keepdim=True)
    probs = probs / torch.sqrt(((probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
    ####
    #probs = probs / torch.sqrt(((all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))
    cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
    pairwise_inds = 1 - torch.eye(n_heads).cuda(device)
    den = bsz * (n_heads -1) * n_heads
    loss = ((cov_mat * pairwise_inds).abs().sum() / den)
    return loss