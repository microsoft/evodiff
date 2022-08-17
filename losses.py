import numpy as np
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer, matrixMul
from dms.collaters import random_sample, sample_transition_matrix

def sample_prior_gaussian(q):
    samples = q.shape[0]
    seq_len = q.shape[1]
    sample_shape = (torch.zeros(1,26)).shape
    m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    prior = torch.zeros(q[:,:,:26].shape)
    print(q.size)
    for i in range(samples):
        for j in range(seq_len):
            aa_prob = m.sample(sample_shape=sample_shape).squeeze()
            aa_prob = aa_prob/aa_prob.sum()
            # print(aa_prob)
            # print(aa_prob.shape)
            # print(aa_prob.sum())
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
                _timesteps = torch.tensor(np.repeat(timesteps[i], timesteps[i], axis=0))  # expand timesteps so dim matches loss dim
                _timesteps = _timesteps.to(tgt.device)
                rwt_term = 1. / _timesteps  # Hoogeboom OARDM
                _n_tokens = torch.repeat_interleave(n_tokens[i], len(loss), axis=0)
                ce_loss = _n_tokens * rwt_term * loss
            if not self.reweight: # Uses lambda reweighting term
                rwt_term = torch.tensor(np.repeat(self._lambda, len(loss), axis=0))
                #print(rwt_term.shape, loss.shape)
                rwt_term = rwt_term.to(tgt.device)
                #print(rwt_term.shape, loss.shape)
                ce_loss = rwt_term * loss
            ce_losses.append(ce_loss.sum()) # reduce mean
            nll_losses += loss.sum()
        nll_losses = nll_losses/mask.sum()
        ce_losses = torch.stack(ce_losses, dim=0).sum()/n_tokens.sum() # divide by D so we can compare to bits per char
        #print(ce_losses, nll_losses)
        return ce_losses, nll_losses.to(torch.float64)


class AustinLoss(KLDivLoss):
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, _lambda=0.01):
        self.tmax = tmax
        self._lambda = _lambda
        self.tokenizer = Tokenizer()
        super().__init__(reduction=reduction, log_target=log_target)
    def forward(self, q, pred, tgt, timestep, Q):
        T_inf = self.tmax
        p = torch.nn.functional.softmax(pred[:,:,:26], dim=2) # ignoring mask/pad
        alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        losses = []
        prior = sample_prior_gaussian(q) # random prior, for absorbing state
        prior = prior.to(tgt.device)
        for i in range(tgt.shape[0]): # enumerate over batch
            print(self.tokenizer.untokenize(tgt[i]))
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = CrossEntropyLoss()
                r_loss = reconstruction_loss(pred[i], tgt[i])
                losses.append(r_loss)
                print("timestep", timestep[i], "seq_len", len(tgt[i]), "r_loss_i", r_loss)
            elif timestep[i] >= T_inf-1:
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                print(prior[i].shape, q[i,:,:26].shape)
                kl_loss_i = super().forward(prior[i].log(), q[i,:,:26]) # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
                #print("timestep", timestep[i], "seq_len", len(tgt[i]), "kINF_loss_i", kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                prob = p[i]
                q_true = q[i,:,:26] # ignoring mask/pad
                x_0_bar = torch.zeros(len(prob))
                x_0_bar = random_sample(x_0_bar, prob, alphabet) # sample x_0_bar from prediction prob
                #print(self.tokenizer.untokenize(x_0_bar))
                x_0_bar = torch.tensor(self.tokenizer.one_hot(x_0_bar, tokenized=True)) # one hot
                x_0_bar = x_0_bar.to(tgt.device)
                # Calculate q given model predictions
                x_t, q_x_t = sample_transition_matrix(x_0_bar, Q[timestep[i]], 1, alphabet)
                x_t = torch.tensor(self.tokenizer.one_hot(x_t, tokenized=True))  # one hot
                x_t = x_t.to(tgt.device)
                p_theta = [] #torch.zeros(q_true.shape)
                for j in range(len(x_0_bar)):  # enumerate over masked tokens in sequence (dim 1xK)
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
                print("timestep", timestep[i], "seq_len", len(tgt[i]), "k_loss_i", kl_loss_i)
                losses.append(kl_loss_i)
        # TODO append loss to CSV w/ timestep for plotting #
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        return losses, lvb

# class AustinLoss(KLDivLoss):
#     def __init__(self, reduction='batchmean', log_target=False, _lambda=0.01, tokenizer=Tokenizer()):
#         self._lambda = _lambda
#         self.tokenizer = Tokenizer()
#         super().__init__(reduction=reduction, log_target=log_target)
#     def forward(self, q, pred, tgt, mask, timestep, Q, input_mask):
#         # Probs of each token need to sum to 1
#         p = torch.nn.functional.softmax(pred, dim=2)
#         alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
#         mask = mask.unsqueeze(-1).bool()
#         mask_tokens = mask.sum(axis=1)  # len batch
#         #Q = torch.tensor(Q)
#         kl_losses = []
#         c = 0 # constant in case seq empty
#         for i in range(tgt.shape[0]): # enumerate over batch
#             prob = torch.masked_select(p[i], mask[i]).view(mask_tokens[i], len(alphabet))  # predictions for each seq
#             q_true = torch.masked_select(q[i], mask[i]).view(mask_tokens[i], len(alphabet))  # predictions for each seq
#             x_0_bar = torch.zeros(len(prob))
#             x_0_bar = random_sample(x_0_bar, prob, alphabet) # sample x_0_bar from predictions
#             #print(self.tokenizer.untokenize(x_0_bar))
#             pad_mask = (x_0_bar != self.tokenizer.pad_id) # Filter mask-> pad transition (nan)
#             mask_mask = (x_0_bar != self.tokenizer.mask_id) # Filter mask-> mask transition (0)
#             pad_mask_mask = pad_mask * mask_mask
#             x_0_bar = torch.masked_select(x_0_bar, pad_mask_mask)
#             #print(self.tokenizer.untokenize(x_0_bar))
#             x_0_bar = torch.tensor(self.tokenizer.one_hot(x_0_bar, tokenized=True)) # one hot
#             x_0_bar = x_0_bar.to(tgt.device)
#             # Calculate q given model predictions
#             x_t, q_x_t = sample_transition_matrix(x_0_bar, Q[timestep[i]], 1, alphabet)
#             x_t = torch.tensor(self.tokenizer.one_hot(x_t, tokenized=True))  # one hot
#             #Q_tminus1 = matrixMul(Q, timestep[i]-1)
#             Q_tminus1 = Q[timestep[i]-1]
#             # move to device
#             x_t = x_t.to(tgt.device)
#             pad_mask_mask = pad_mask_mask.to(tgt.device)
#             # initiate for saving
#             #print(prob, q_true)
#             kl_loss_i = []
#             if len(x_0_bar) > 0: # For short sequences it is likely in early training all tokens predicted to be mask or pad
#                 prob = prob[pad_mask_mask]  # Filter mask-> pad transitions from prob for p_theta calculation
#                 q_true = q_true[pad_mask_mask]  # Filter mask-> pad transitions from prob for kl calculation
#                 for j in range(len(x_0_bar)): # enumerate over masked tokens in sequence (dim 1xK)
#                     # Calculate q(x_t-1 | x_t, x_0_bar) - eq 3
#                     # A = x_t * torch.transpose(Q_t) (shape - 1 x K)
#                     A = torch.matmul(x_t[j].unsqueeze(0), torch.t(Q[timestep[i]]))
#                     #print("A", A)
#                     # B = x_0_bar * Q_t-1 (shape - 1 x K)
#                     B = torch.matmul(x_0_bar[j].unsqueeze(0), Q_tminus1)
#                     #print("B", B)
#                     num = torch.mul(A, B)  # element wise (shape 1 x K)
#                     den = (torch.t(x_t[j]))  # (shape K x 1) # TODO:ask if this ok, 1/x_t gives inf for one hot encoded (?)
#                     #print("den", den)
#                     q_t_j = torch.matmul(num, den)  # shape 1x1
#                     # Calculate p_theta_j
#                     p_theta_j = q_t_j * prob[j]
#                     p_theta_j = p_theta_j/p_theta_j.sum() # renormalize; sum prob to 1
#                     #print(p_theta_j.sum(), q_true[j].sum())
#                     kl_j = super().forward(p_theta_j.log(), q_true[j]) # KLDivLoss expects input in log-space
#                     #print("klj loss", kl_j)
#                     kl_loss_i.append(kl_j)
#                 kl_loss_i = torch.stack(kl_loss_i).sum() # loss per seq
#                 #print("kli", kl_loss_i)
#                 kl_losses.append(kl_loss_i)
#             else: # if empty sequence ignore
#                 c += 1 # subtract seq from batch normalization
#         #print(len(kl_losses), tgt.shape[0])
#         kl_losses = (torch.stack(kl_losses, dim=0).sum()/(tgt.shape[0]-c)) # loss per batch, norm by batchsize
#         #print("kl", kl_losses, kl_losses.dtype)
#         return kl_losses