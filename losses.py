import numpy as np
import torch
from torch.nn import CrossEntropyLoss

class MaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence, using a reweighting term for diffusion
    models 1/(D-t+1) output in OAMaskCollater
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
    """

    def __init__(self, weight=None, reduction='none'):
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        # Select
        n = mask.sum()
        p = torch.masked_select(pred, mask).view(n, -1) # predictions for each mask
        t = torch.masked_select(tgt, mask.squeeze())
        loss = super().forward(p,t)
        # Dot prod loss w/ reweighting term
        timesteps = np.repeat(timesteps, timesteps, axis=0) # expand timesteps so dim matches loss
        timesteps = torch.tensor(timesteps, dtype=torch.float64)
        timesteps = timesteps.to(loss.device)
        rwt_term = 1. / timesteps  # Hoogeboom OARDM
        print(rwt_term.shape, loss.shape)
        loss_rwt = torch.dot(rwt_term, loss.to(torch.float64)) # rwt has dim of batch size, loss is meaned over all mask
        print(loss_rwt)
        return loss_rwt