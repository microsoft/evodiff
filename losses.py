import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer, matrixMul
from dms.collaters import sample_transition_matrix
from dms.constants import ALL_AAS

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
    one_hot: batchsize x seq_len x n_tokens(ALL_AAS)
    input_mask: bool of non-padded locations
    """
    def __init__(self, weight=None, reduction='none', tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, one_hot, input_mask):
        p = pred[:, :, :len(ALL_AAS)]
        nonpad_loc = input_mask.sum(axis=1)
        ce_losses = 0
        for i in range(p.shape[0]): # iterate over batchsize
            D = int(nonpad_loc[i].item()) # index non-pad entries
            ce_loss = super().forward(p[i, :D, :], one_hot[i, :D, :])
            ce_losses += ce_loss.sum()
        ce_losses = ce_losses/nonpad_loc.sum()
        return ce_losses

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

    def forward(self, src, q, q_minus1, pred, one_hot, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(pred[:, :, :self.len_aa], dim=2) # ignoring mask/pad
        losses = []
        nonpad_loc = input_mask.sum(axis=1)
        for i in range(one_hot.shape[0]): # enumerate over batch
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = D3PMCELoss()
                r_loss = reconstruction_loss(pred[i].unsqueeze(0), one_hot[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                losses.append(r_loss)
            elif timestep[i] == self.tmax:
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                D = int(nonpad_loc[i].item()) # want prior/q in shape of seq len (q has shape of longest seq in batch)
                q_temp = q[i, :D, :]
                prior = sample_prior(q_temp.shape[0], q_temp.shape[1], all_aas=self.tokenizer.all_aas)
                prior = prior.to(one_hot.device)
                kl_loss_i = super().forward(q_temp.log(), prior)  # fKLDivLoss expects input in log-space
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                D = int(nonpad_loc[i]) # non pad locations
                prob = p[i, :D]
                q_true = q_minus1[i,:D]
                src_i = src[i, :D]
                p_theta_marg = torch.zeros((prob.shape[0], prob.shape[1]), dtype=torch.double)
                for j in range(prob.shape[0]): # iterate over each residue in sequence
                    # Compute p_theta(x_tminus1| x_t)
                    #cat = torch.linspace(0,prob.shape[1]-1, prob.shape[1], dtype=torch.int64)
                    #x_0_bar = self.tokenizer.one_hot(cat) # one hot encoding of all tokens [K x K]
                    #x_0_bar = x_0_bar.to(one_hot.device)
                    x_t = self.tokenizer.one_hot(src_i[j]) # one_hot encoding of the true corrupted sequence [1 x K]
                    A = torch.matmul(x_t, torch.t(Q[timestep[i]])) # [1 x K]
                    B = Q_bar[timestep[i]-1]
                    #B = torch.matmul(x_0_bar, Q_bar[timestep[i] - 1]) # [K x K]
                    q_t = torch.mul(A, B) # [K x K]
                    p_theta = q_t * prob[j] # [ K x K]
                    p_theta_marg_j = p_theta.sum(axis=1) # sum over cols to marginalize each token at the same time
                    p_theta_marg[j] = p_theta_marg_j / p_theta_marg_j.sum() # normalize token probabilities
                    #p_theta_marg[j] = p_theta_marg_j
                    #import pdb; pdb.set_trace()
                p_theta_marg = p_theta_marg.to(one_hot.device)
                kl_loss_i = super().forward(q_true.log(), p_theta_marg)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (one_hot.shape[0]))  # loss per batch, norm by batchsize
        return lvb


class D3PMCELossMSA(CrossEntropyLoss):
    """
    Standard cross entropy loss
    Wrapped to deal with padding and normalize by # of non-padded locations
    pred: batchsize x seq_len x n_tokens(PROTEIN_ALPHABET)
    one_hot: batchsize x seq_len x n_tokens(ALL_AAS)
    input_mask: bool of non-padded locations
    """
    def __init__(self, weight=None, reduction='none', tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, one_hot, input_mask):
        p = pred[:, :, :, :len(self.tokenizer.all_aas)]
        nonpad_loc = input_mask.sum(axis=(1,2))
        ce_losses = 0
        for i in range(p.shape[0]): # iterate over batchsize
            D = int(nonpad_loc[i])
            p_current = p[i, :, :D, :]
            t_current = one_hot[i, :, :D, :]
            p_current = p_current.flatten(start_dim=0, end_dim=1)
            t_current = t_current.flatten(start_dim=0, end_dim=1)
            ce_loss = super().forward(p_current, t_current)
            ce_losses += ce_loss.sum()
        ce_losses = ce_losses/nonpad_loc.sum()
        print("ce", ce_losses) # TODO why is CE loss so large?
        return ce_losses


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

    def forward(self, src, q, q_minus1, pred, one_hot, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(pred[:, :, :, :self.len_aa], dim=3) # ignoring mask/pad
        losses = []
        nonpad_loc = input_mask.sum(axis=(2))
        for i in range(one_hot.shape[0]): # enumerate over batch
            if timestep[i] <= 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = D3PMCELossMSA(tokenizer=self.tokenizer)
                r_loss = reconstruction_loss(pred[i].unsqueeze(0), one_hot[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                losses.append(r_loss)
            elif timestep[i] == self.tmax:
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                D = int(nonpad_loc[i, :]) # want prior/q in shape of seq len (q has shape of longest seq in batch)
                q_temp = q[i, :, :D, :]
                prior = sample_prior3D(q_temp.shape[0], q_temp.shape[1], q_temp.shape[2], all_aas=self.tokenizer.all_aas)
                prior = prior.to(one_hot.device)
                kl_loss_i = super().forward(q_temp.log(), prior)  # fKLDivLoss expects input in log-space
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta_marg)
                D = int(nonpad_loc[i][0]) # all seq in one MSA are padded to the same length, use first seq as ref
                prob = p[i, :, :D].flatten(start_dim=0, end_dim=1) # [pos x tokens]
                q_true = q_minus1[i, :, :D].flatten(start_dim=0, end_dim=1)
                src_i = src[i, :, :D].flatten(start_dim=0, end_dim=1)
                p_theta_marg = torch.zeros((prob.shape[0], prob.shape[1]), dtype=torch.double)
                for j in range(prob.shape[0]):  # iterate over each residue in sequence
                    # Compute p_theta(x_tminus1| x_t)
                    # cat = torch.linspace(0, prob.shape[1] - 1, prob.shape[1], dtype=torch.int64)
                    # x_0_bar = self.tokenizer.one_hot(cat)  # one hot encoding of all tokens [K x K]
                    # x_0_bar = x_0_bar.to(one_hot.device)
                    x_t = self.tokenizer.one_hot(src_i[j])  # one_hot encoding of x_t at position j [1 x K]
                    A = torch.matmul(x_t, torch.t(Q[timestep[i]]))  # [1 x K]
                    B = Q_bar[timestep[i]-1]
                    #B = torch.matmul(x_0_bar, Q_bar[timestep[i] - 1])  # [K x K]
                    q_t = torch.mul(A, B)  # [K x K]
                    p_theta = q_t * prob[j]  # [ K x K]
                    p_theta_marg_j = p_theta.sum(axis=1)  # sum over rows to marginalize each token at the same time
                    p_theta_marg[j] = p_theta_marg_j / p_theta_marg_j.sum()  # normalize token probabilities
                p_theta_marg = p_theta_marg.to(one_hot.device)
                kl_loss_i = super().forward(q_true.log(), p_theta_marg)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (one_hot.shape[0]))  # loss per batch, norm by batchsize
        return lvb