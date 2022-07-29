import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer
from dms.collaters import random_sample, sample_transition_matrix, matrixMul
import os

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
    def __init__(self, weight=None, reduction='none', reweight=True, _lambda=0.01):
        self.reweight=reweight
        self._lambda=_lambda
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps, input_mask):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
            input_mask = input_mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        input_mask = input_mask.bool() # padded seq
        # Select
        #n = mask.sum() # len mask tokens
        mask_tokens = mask.sum(axis=1) # len batch
        n_tokens = input_mask.sum(axis=1) # len batch
        nll_losses = 0
        ce_losses = []
        for i in range(tgt.shape[0]): # iterate over each sequence in batch
            p = torch.masked_select(pred[i], mask[i]).view(mask_tokens[i], -1) # predictions for each mask
            t = torch.masked_select(tgt[i], mask[i].squeeze())#.squeeze())
            loss = super().forward(p, t)
            #print("loss", loss)
            if self.reweight:
                # Reweight for summation over t
                _timesteps = torch.tensor(np.repeat(timesteps[i], timesteps[i], axis=0), dtype=torch.float64)  # expand timesteps so dim matches loss dim
                _timesteps = _timesteps.to(tgt.device)
                rwt_term = 1. / _timesteps  # Hoogeboom OARDM
                _n_tokens = torch.repeat_interleave(n_tokens[i], len(loss), axis=0)
                ce_loss = _n_tokens * rwt_term * loss
            if not self.reweight:
                rwt_term = torch.tensor(np.repeat(self._lambda, len(loss), axis=0), dtype=torch.float64)
                #print(rwt_term.shape, loss.shape)
                rwt_term = rwt_term.to(tgt.device)
                #print(rwt_term.shape, loss.shape)
                ce_loss = rwt_term * loss
            ce_losses.append(ce_loss.sum()) # reduce mean
            nll_losses += loss.sum()
        nll_losses = nll_losses/mask.sum()
        ce_losses = torch.stack(ce_losses, dim=0).sum()/n_tokens.sum() # divide by D so we can compare to bits per char
        #print(ce_losses, nll_losses)
        return ce_losses, nll_losses

class AustinLoss(KLDivLoss):
    def __init__(self, reduction='batchmean', log_target=False, _lambda=0.01):
        self._lambda = _lambda
        self.tokenizer = Tokenizer()
        super().__init__(reduction=reduction, log_target=log_target)
    def forward(self, q, pred, tgt, mask, timestep, Q, input_mask):
        # Probs of each token need to sum to 1
        p = torch.nn.functional.softmax(pred, dim=2)
        alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        mask = mask.unsqueeze(-1).bool()
        mask_tokens = mask.sum(axis=1)  # len batch
        Q = torch.tensor(Q)
        kl_losses = []
        for i in range(tgt.shape[0]): # enumerate over batch
            # Select p and q for masked tokens
            prob = torch.masked_select(p[i], mask[i]).view(mask_tokens[i], -1)  # predictions for each seq
            q_true = torch.masked_select(q[i], mask[i]).view(mask_tokens[i], -1)  # predictions for each seq
            x_0_bar = torch.zeros(len(prob))
            x_0_bar = random_sample(x_0_bar, prob, alphabet) # sample x_0_bar from predictions
            pad_mask = (x_0_bar != self.tokenizer.pad_id) # Filter mask-> pad transition (nan)
            x_0_bar = torch.masked_select(x_0_bar, pad_mask).view(pad_mask[i], -1).squeeze()
            x_0_bar = torch.tensor(self.tokenizer.one_hot(x_0_bar, tokenized=True)) # one hot
            # Calculate q given model predictions
            x_t, q_x_t = sample_transition_matrix(x_0_bar, Q, timestep[i], alphabet)
            x_t = torch.tensor(self.tokenizer.one_hot(x_t, tokenized=True))  # one hot
            Q_tminus1 = matrixMul(Q, timestep[i]-1)
            # move to device
            x_0_bar = x_0_bar.to(tgt.device)
            x_t = x_t.to(tgt.device)
            Q = Q.to(tgt.device)
            Q_tminus1 = Q_tminus1.to(tgt.device)
            pad_mask = pad_mask.to(tgt.device)
            # initiate for saving
            prob = prob[pad_mask] # Filter mask-> pad transitions from prob for p_theta calculation
            q_true = q_true[pad_mask] # Filter mask-> pad transitions from prob for kl calculation
            print("prob/q shape", prob.shape, q_true.shape)
            kl_loss_i = []
            for j in range(len(x_0_bar)): # enumerate over masked tokens in sequence (dim 1xK)
                # Calculate q(x_t-1 | x_t, x_0_bar) - eq 3
                # A = x_t * torch.transpose(Q_t) (shape - 1 x K)
                A = torch.matmul(x_t[j].unsqueeze(0), torch.t(Q))
                #print("A", A)
                # B = x_0_bar * Q_t-1 (shape - 1 x K)
                B = torch.matmul(x_0_bar[j].unsqueeze(0), Q_tminus1)
                #print("B", B)
                num = torch.mul(A, B)  # element wise (shape 1 x K)
                #print("num", num)
                den = (torch.t(x_t[j]))  # (shape K x 1) # TODO: ASK 1/one_hot gives inf for one hot encoded (?)
                #print("den", den)
                # Calc q = num/den
                q_t_j = torch.matmul(num, den)  # shape 1x1
                # Calculate p_theta_j
                p_theta_j = q_t_j * prob[j]
                p_theta_j = p_theta_j/p_theta_j.sum() # renormalize sum to 1
                #print(p_theta_j.sum(), q_true[j].sum())
                kl_j = super().forward(p_theta_j.log(), q_true[j]) # KLDivLoss expects input in log-space
                #print("klj loss", kl_j)
                kl_loss_i.append(kl_j)
            kl_loss_i = torch.stack(kl_loss_i, dim=0).sum() # loss per seq
            #print("kli", kl_loss_i)
            kl_losses.append(kl_loss_i)
        kl_losses = torch.stack(kl_losses, dim=0).sum() # loss per batch
        #print("kl", kl_losses)
        return kl_losses