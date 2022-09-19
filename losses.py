import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer, matrixMul
from dms.collaters import sample_transition_matrix
from dms.constants import ALL_AAS
from torch.nn.functional import normalize

def sample_prior(a,b, all_aas=ALL_AAS):
    """
    Returns prior for KL at T-> inf with same shape as q over total possible values (all_aas)
    Prior is a stationary distribution; uniform distribution over number of values
    """
    prior = torch.empty(a,b)
    prior = torch.ones_like(prior) / len(all_aas)
    return prior

def sample_prior3D(a,b,c, all_aas=ALL_AAS):
    """
    Returns prior for KL at T-> inf with same shape as q over total possible values (all_aas)
    Prior is a stationary distribution; uniform distribution over number of values
    """
    prior = torch.empty(a,b,c)
    prior = torch.ones_like(prior) / len(all_aas)
    return prior

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
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, mask, timesteps, input_mask):
        alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
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
            p = torch.masked_select(pred[i], mask[i]).view(mask_tokens[i], len(alphabet)) # predictions for each mask
            t = torch.masked_select(tgt[i], mask[i].squeeze())#.squeeze())
            loss = super().forward(p, t)
            #print("loss", loss)
            if self.reweight: # Uses autoreg reweighting term
                # Reweight for summation over t
                _timesteps = timesteps[i].repeat_interleave(timesteps[i])
                rwt_term = 1. / _timesteps  # Hoogeboom OARDM
                _n_tokens = torch.repeat_interleave(n_tokens[i], len(loss), axis=0)
                ce_loss = _n_tokens * rwt_term * loss
            if not self.reweight:  # For D3PM reweight in train loop
                ce_loss = loss
            ce_losses.append(ce_loss.sum()) # reduce mean
            nll_losses += loss.sum()
        nll_losses = nll_losses/mask.sum()
        ce_losses = torch.stack(ce_losses, dim=0).sum()/n_tokens.sum() # divide by D so we can compare to bits per char
        return ce_losses, nll_losses.to(torch.float64)


class D3PMCELoss(CrossEntropyLoss):
    """
    Standard cross entropy loss
    Wrapped to deal with padding and normalize by # of non-padded locations
    pred: batchsize x seq_len x n_tokens(PROTEIN_ALPHABET)
    tgt: batchsize x seq_len
    input_mask: bool of non-padded locations
    """
    def __init__(self, weight=None, reduction='mean', tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, input_mask):
        p = pred[:, :, :len(self.tokenizer.all_aas)]
        batch, length, tokens = p.shape
        nonpad_loc = input_mask.bool()
        p_unpadded = torch.masked_select(p,nonpad_loc.unsqueeze(-1).expand(p.shape))
        p_unpadded = p_unpadded.reshape(-1, tokens)
        t_unpadded = torch.masked_select(tgt, nonpad_loc)
        ce_loss = super().forward(p_unpadded, t_unpadded)
        return ce_loss

class D3PMLVBLoss(KLDivLoss):
    """
    Lower variational bound loss as defined in Austin et al.
        Shape:
            Inputs:
                - q: (N, L, n_tokens) forward prob dist
                - pred: (N, L, n_tokens) predicted reverse dist
                - tgt: (N, L)
                - timestep (N)
                - Q (n_tokens x n_tokens) transition matrix

        # Returns
        """
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer()):
        self.tmax = tmax
        self.tokenizer = tokenizer
        self.len_aa = len(self.tokenizer.all_aas)
        super().__init__(reduction=reduction, log_target=log_target)

    def forward(self, src, q, q_minus1, predictions, tgt, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predictions[:, :, :self.len_aa], dim=2) # ignoring mask/pad
        losses = []
        nonpad_loc = input_mask.sum(axis=1)
        for i in range(tgt.shape[0]): # enumerate over batch
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = D3PMCELoss()
                r_loss = reconstruction_loss(predictions[i].unsqueeze(0), tgt[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                losses.append(r_loss)
            elif timestep[i] == self.tmax: # Not needed to compute gradients
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                D = int(nonpad_loc[i].item()) # want prior/q in shape of seq len (q has shape of longest seq in batch)
                q_true = q[i, :D]
                prior = sample_prior(q_true.shape[0], q_true.shape[1], all_aas=self.tokenizer.all_aas)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q_true)  # fKLDivLoss expects input in log-space
                #print("KL SHOULD BE ~ZERO", kl_loss_i)
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                D = int(nonpad_loc[i]) # non pad locations
                pred = p[i, :D]
                q_true_minus1 = q_minus1[i,:D]
                x_t_tokenized = src[i, :D]
                x_t = self.tokenizer.one_hot(x_t_tokenized)
                #x_t = q[i,:D]
                A = torch.mm(x_t, torch.t(Q[timestep[i]])) # [P x K]
                B = Q_bar[timestep[i]-1] # [K x K]
                q_t = torch.mul(A.unsqueeze(1), B) # [P x K x K]
                pred = pred.to(torch.float64) # must use 64 not 32 or p_theta_marg
                p_theta_marg = torch.bmm(q_t, pred.unsqueeze(2)).squeeze() # [P x K] this marginalizes over dim=2
                p_theta_marg = p_theta_marg/p_theta_marg.sum(axis=1, keepdim=True) # normalize probabilities at each position
                p_theta_marg = p_theta_marg.to(tgt.device)
                kl_loss_i = super().forward(p_theta_marg.log(), q_true_minus1)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        print(lvb)
        return lvb


class D3PMCELossMSA(CrossEntropyLoss):
    """
    Standard cross entropy loss
    Wrapped to deal with padding and normalize by # of non-padded locations
    pred: batchsize x seq_len x n_tokens(PROTEIN_ALPHABET)
    one_hot: batchsize x seq_len x n_tokens(ALL_AAS)
    input_mask: bool of non-padded locations
    """
    def __init__(self, weight=None, reduction='mean', tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, input_mask):
        p = pred[:, :, :, :len(self.tokenizer.all_aas)]
        batchsize, length, depth, tokens = p.shape
        nonpad_loc = input_mask.bool()
        p_unpadded = torch.masked_select(p, nonpad_loc.unsqueeze(-1).expand(p.shape))
        p_unpadded = p_unpadded.reshape(-1, tokens)
        t_unpadded = torch.masked_select(tgt, nonpad_loc)
        ce_loss = super().forward(p_unpadded, t_unpadded)
        #[print("ce", ce_loss) # TODO why is CE loss so large?
        return ce_loss


class D3PMLVBLossMSA(KLDivLoss):
    """
    Lower variational bound loss as defined in Austin et al.
        Shape:
            Inputs:
                - q: (N, L, n_tokens) forward prob dist
                - pred: (N, L, n_tokens) predicted reverse dist
                - tgt: (N, L)
                - timestep (N)
                - Q (n_tokens x n_tokens) transition matrix

        Returns
        """
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer()):
        self.tmax = tmax
        self.tokenizer = tokenizer
        self.len_aa = len(self.tokenizer.all_aas)
        super().__init__(reduction=reduction, log_target=log_target)

    def forward(self, src, one_hot, q, q_minus1, predictions, tgt, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predictions[:, :, :, :self.len_aa], dim=3) # ignoring mask/pad
        losses = []
        nonpad_loc = input_mask.sum(axis=2)
        for i in range(len(tgt)): # enumerate over batch
            D = int(nonpad_loc[i][0])  # all seq in one MSA are padded to the same length, use first seq as ref
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = D3PMCELossMSA(tokenizer=self.tokenizer)
                r_loss = reconstruction_loss(predictions[i].unsqueeze(0), tgt[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                #print(r_loss)
                losses.append(r_loss)
            elif timestep[i] == self.tmax:  # Not needed to compute gradients
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                q_true = q[i, :, :D, :]
                prior = sample_prior3D(q_true.shape[0], q_true.shape[1], q_true.shape[2], all_aas=self.tokenizer.all_aas)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q_true)  # fKLDivLoss expects input in log-space
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta_marg)
                pred = p[i, :, :D].flatten(start_dim=0, end_dim=1) # [pos x tokens]
                q_true_minus1 = q_minus1[i, :, :D].flatten(start_dim=0, end_dim=1)
                x_t = one_hot[i, :, :D].flatten(start_dim=0, end_dim=1)
                A = torch.mm(x_t, torch.t(Q[timestep[i]]))  # [P x K]
                B = Q_bar[timestep[i] - 1]  # [K x K]
                q_t = torch.mul(A.unsqueeze(1), B)  # confirmed this is the same as for loop
                pred = pred.to(torch.float64)  # must use 64 not 32 or p_theta_marg
                p_theta_marg = torch.bmm(q_t, pred.unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1,keepdim=True)  # normalize probabilities at each position
                p_theta_marg = p_theta_marg.to(tgt.device)
                kl_loss_i = super().forward(p_theta_marg.log(), q_true_minus1)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        return lvb