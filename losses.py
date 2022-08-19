import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer, matrixMul
from dms.collaters import random_sample, sample_transition_matrix
from dms.constants import ALL_AAS

def sample_prior_gaussian(q, all_aas=ALL_AAS):
    samples = q.shape[0]
    seq_len = q.shape[1]
    sample_shape = (torch.zeros(1,len(all_aas))).shape
    m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    prior = torch.zeros(q[:, :, :len(all_aas)].shape)
    print(q.size)
    for i in range(samples):
        for j in range(seq_len):
            aa_prob = m.sample(sample_shape=sample_shape).squeeze()
            aa_prob = aa_prob/aa_prob.sum()
            prior[i,j] = aa_prob
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
    def __init__(self, weight=None, reduction='none', reweight=True, _lambda=0.01, tokenizer=Tokenizer()):
        self.reweight=reweight
        self._lambda=_lambda
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
            else: # D3PM omits reweighting term
                ce_loss = loss
            ce_losses.append(ce_loss.sum()) # reduce mean
            nll_losses += loss.sum()
        nll_losses = nll_losses/mask.sum()
        ce_losses = torch.stack(ce_losses, dim=0).sum()/n_tokens.sum() # divide by D so we can compare to bits per char
        return ce_losses, nll_losses.to(torch.float64)


class LVBLoss(KLDivLoss):
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
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, _lambda=0.01, all_aas=ALL_AAS):
        self.tmax = tmax
        self._lambda = _lambda
        self.tokenizer = Tokenizer()
        self.len_aa = len(all_aas)
        super().__init__(reduction=reduction, log_target=log_target)
    def forward(self, q, pred, tgt, timestep, Q):
        p = torch.nn.functional.softmax(pred[:, :, :self.len_aa], dim=2) # ignoring mask/pad
        alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        losses = []
        prior = sample_prior_gaussian(q) # random prior, for absorbing state
        prior = prior.to(tgt.device)
        for i in range(tgt.shape[0]): # enumerate over batch
            #print(self.tokenizer.untokenize(tgt[i]))
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = CrossEntropyLoss()
                r_loss = reconstruction_loss(pred[i], tgt[i])
                losses.append(r_loss)
            elif timestep[i] == self.tmax-1:
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                kl_loss_i = super().forward(prior[i].log(), q[i, :, :self.len_aa]) # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                prob = p[i]
                print(q.shape)
                q_true = q[i, :, :self.len_aa] # ignoring mask/pad
                # sample x_0_bar from predicted prob
                x_0_bar = random_sample(torch.zeros(len(prob)), prob, alphabet)
                x_0_bar = torch.tensor(self.tokenizer.one_hot(x_0_bar, tokenized=True)) # one hot
                x_0_bar = x_0_bar.to(tgt.device)
                # Calculate q(forward) given model predictions
                x_t, q_x_t = sample_transition_matrix(x_0_bar, Q[timestep[i]], 1, alphabet)
                x_t = torch.tensor(self.tokenizer.one_hot(x_t, tokenized=True))  # one hot
                x_t = x_t.to(tgt.device)
                p_theta = []
                for j in range(len(x_0_bar)):  # enumerate over tokens in sequence (dim 1xK)
                    # A = x_t * torch.transpose(Q_t) (shape - 1 x K)
                    A = torch.matmul(x_t[j].unsqueeze(0), torch.t(Q[timestep[i]]))
                    #print("A", A.shape, A)
                    # B = x_0_bar * Q_t-1 (shape - 1 x K)
                    B = torch.matmul(x_0_bar[j].unsqueeze(0), Q[timestep[i-1]])
                    #print("B", B.shape, B)
                    q_t_j = torch.mul(A, B)  # element wise (shape 1 x K)
                    p_theta_j = q_t_j * prob[j]
                    p_theta_j = p_theta_j / p_theta_j.sum()  # renormalize; sum prob to 1
                    p_theta.append(p_theta_j.squeeze())
                p_theta = torch.stack(p_theta)
                p_theta = p_theta.to(tgt.device)
                kl_loss_i = super().forward(p_theta.log(), q_true)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        # TODO: remove this append loss to CSV w/ timestep for plotting #
        losses = torch.stack(losses) # for plotting purposes only - remove this line
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        return losses, lvb