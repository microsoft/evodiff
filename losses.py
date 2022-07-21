import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss

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
    def __init__(self, weight=None, reduction='none', reweight=True):
        self.reweight=reweight
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps):
        # Make sure we have that empty last dimension
        #print("pred, mask, tgt shape", pred.shape, mask.shape, tgt.shape)
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        # Select
        n = mask.sum()
        #print("n",n)
        #print("pred, mask, tgt shape", pred.shape, mask.shape, tgt.shape)
        p = torch.masked_select(pred, mask).view(n, -1) # predictions for each mask
        t = torch.masked_select(tgt, mask.squeeze())
        loss = super().forward(p, t)
        if self.reweight == True:
            # Reweight for summation over t
            timesteps = np.repeat(timesteps, timesteps, axis=0) # expand timesteps so dim matches loss dim
            timesteps = torch.tensor(timesteps, dtype=torch.float64)
            timesteps = timesteps.to(loss.device)
            rwt_term = 1. / timesteps  # Hoogeboom OARDM
            loss = torch.dot(rwt_term, loss.to(torch.float64))
        else:
            _lambda = 0.01  # TODO fix this
            rwt_term = torch.tensor(np.repeat(_lambda, loss.shape[0], axis=0), dtype=torch.float64)
            rwt_term = rwt_term.to(loss.device)
            loss = torch.dot(rwt_term, loss.to(torch.float64))
        return loss

class AustinLoss(KLDivLoss):
    def __init__(self, reduction='batchmean'):
        super().__init__(reduction=reduction)
    def forward(self, q, p, tgt, mask, timestep):
        # KL divergence between q and p
        p_norm = torch.nn.functional.softmax(p[:,:,0:26], dim=2) # janky way to ignore specials char here - TODO: trouble shoot later
        q = q[:,:,0:26]
        #print("p", p.shape, "q", q.shape)
        #print(p[0][0], "SUM", p[0][0].sum(), p[0][0].shape) # p/q probs must sum to 1
        #print(q[0][0], "SUM", q[0][0].sum(), q[0][0].shape)
        elbo_loss = super().forward(p_norm,q) #input, target (note: reverse notation of documentation b/c of DM model notation)
        #print(elbo_loss)
        # Negative cross entropy
        ce = MaskedCrossEntropyLoss(reweight=False)
        loss_ce = ce(p, tgt, mask, timestep)
        #print(loss_ce)

        # loss = -elbo.mean() + ce_term(lambda?) * ce.mean() # FROM austin github
        loss = elbo_loss + loss_ce
        # loss = -elbo.mean() + lambda(=0.01)*MaskedCrossEntropyLoss(reweight=True)
        return loss