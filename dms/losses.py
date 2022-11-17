import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from dms.utils import Tokenizer
from sequence_models.constants import MSA_AAS
from torch.nn.functional import normalize

def sample_prior(a,b, _len=len(MSA_AAS)):
    """
    Returns prior for KL at T-> inf with same shape as q over total possible values (all_aas)
    Prior is a stationary distribution; uniform distribution over number of values
    """
    prior = torch.empty(a,b)
    prior = torch.ones_like(prior) / _len
    return prior

def sample_prior3D(a,b,c, _len=MSA_AAS):
    """
    Returns prior for KL at T-> inf with same shape as q over total possible values (all_aas)
    Prior is a stationary distribution; uniform distribution over number of values
    """
    prior = torch.empty(a,b,c)
    prior = torch.ones_like(prior) / len(_len)
    return prior

class OAMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
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
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
            input_mask = input_mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        input_mask = input_mask.bool() # padded seq
        # Select
        mask_tokens = mask.sum() # masked tokens
        nonpad_tokens = input_mask.sum(dim=1) # nonpad tokens
        p = torch.masked_select(pred, mask).view(mask_tokens, -1) # [T x K] predictions for each mask char
        t = torch.masked_select(tgt, mask.squeeze()) # [ T ] true mask char
        loss = super().forward(p, t) # [ T ] loss per mask char
        # Calculate reweighted CE loss and NLL loss
        nll_losses = loss.sum()
        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            rwt_term = 1. / timesteps
            rwt_term = rwt_term.repeat_interleave(timesteps)
            _n_tokens = nonpad_tokens.repeat_interleave(timesteps)
            ce_loss = _n_tokens * rwt_term * loss
            ce_losses = ce_loss.sum()  # reduce mean
        else:
            ce_losses = nll_losses
        return ce_losses, nll_losses.to(torch.float64) # normalize by # of tokens


class D3PMCELoss(CrossEntropyLoss):
    """
    Standard cross entropy loss
    Wrapped to deal with padding and normalize by # of non-padded locations
    pred: (N x L x n_tokens) (PROTEIN_ALPHABET)
    tgt: (N x L)
    input_mask: bool of non-padded locations
    """
    def __init__(self, weight=None, reduction='mean', tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)
    def forward(self, pred, tgt, input_mask):
        p = pred[:, :, :self.tokenizer.K]
        batch, length, tokens = p.shape
        nonpad_loc = input_mask.bool()
        p_unpadded = torch.masked_select(p, nonpad_loc.unsqueeze(-1).expand(p.shape))
        p_unpadded = p_unpadded.reshape(-1, tokens)
        t_unpadded = torch.masked_select(tgt, nonpad_loc)
        ce_loss = super().forward(p_unpadded, t_unpadded)
        return ce_loss # mean for entire batch

class D3PMLVBLoss(KLDivLoss):
    """
    Lower variational bound loss as defined in Austin et al.
        Shape:
            Inputs:
                - q: (N, L, n_tokens) forward prob dist at time t
                - q_minus1: (N, L, n_tokens) forward prob dist at time t-1
                - predictions: (N, L, n_tokens) predicted reverse dist (model out)
                - tgt: (N, L) true x_0 sequence
                - input mask (N, L) non-pad locations bool
                - timestep (N)
                - Q (n_tokens x n_tokens) transition matrix
                - Q_bar (n_tokens x n_tokens) cum product of trans matrix at time t

        # Returns
        """
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, tokenizer=Tokenizer()):
        self.tmax = tmax
        self.tokenizer = tokenizer
        self.K = self.tokenizer.K
        super().__init__(reduction=reduction, log_target=log_target)

    def forward(self, src, src_onehot, q, predictions, tgt, tgt_onehot, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predictions[:, :, :self.K], dim=2) # ignoring specials
        losses = []
        nonpad_loc = input_mask.sum(axis=1)
        for i in range(tgt.shape[0]): # enumerate over batch
            D = int(nonpad_loc[i].item())  # want prior/q in shape of seq len (q has shape of longest seq in batch)
            if timestep[i] == 1:
                # CE (L_t=0)
                reconstruction_loss = D3PMCELoss()
                r_loss = reconstruction_loss(predictions[i].unsqueeze(0), tgt[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                losses.append(r_loss)
            elif timestep[i] == self.tmax-1: # Not needed to compute gradients
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                q_true = q[i, :D]
                prior = sample_prior(q_true.shape[0], q_true.shape[1], _len=self.K)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q_true)  # fKLDivLoss expects input in log-space
                #print("KL SHOULD BE ~ZERO", kl_loss_i)
                losses.append(kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                pred = p[i, :D]
                pred = pred.to(torch.float64)
                x_t = src_onehot[i,:D]
                x_0 = tgt_onehot[i,:D]

                # Calculate p_theta_marg, simplified for loops
                A = torch.mm(x_t, torch.t(Q[timestep[i]]))  # [P x K]
                #A_expand = A.unsqueeze(1).expand(A.shape[0], self.K, self.K)  # [P x K x K] # this is the same as A.unsqueeze(1), B_expand)
                B = torch.mm(x_0, Q_bar[timestep[i] - 1]) # P x K
                Q_expand = Q_bar[timestep[i] - 1].unsqueeze(0).expand(A.shape[0], self.K, self.K)  # [ P x K x K]
                B_pred = torch.mul(pred.unsqueeze(2), Q_expand)

                # print("b_pred pkk", B_pred.shape)
                # print(pred.unsqueeze(2)[0][0], Q_expand[0][0], B_pred[0][0])
                # import pdb; pdb.set_trace()
                #B_expand = B.unsqueeze(1).expand(A.shape[0], self.K, self.K) # [ P x K x K]

                q_t = torch.mul(A.unsqueeze(1), B_pred)
                p_theta_marg = torch.bmm(q_t, pred.unsqueeze(2)).squeeze() # mul and sum over model logits

                # calculate q_t_minus_1, simplified for loops
                num = torch.mul(A, B)
                denom = torch.bmm(torch.mm(x_0, Q_bar[timestep[i]]).unsqueeze(1), x_t.unsqueeze(2))
                q_t_minus1 = num / denom.squeeze().unsqueeze(1)

                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)  # re-normalize probs per residue
                p_theta_marg = p_theta_marg.to(tgt.device)
                kl_loss_i = super().forward(p_theta_marg.log(), q_t_minus1)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
        losses = torch.stack(losses) # loss per sequence in batch
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
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
        p = pred[:, :, :, :self.tokenizer.K]
        batchsize, length, depth, tokens = p.shape
        nonpad_loc = input_mask.bool()
        p_unpadded = torch.masked_select(p, nonpad_loc.unsqueeze(-1).expand(p.shape))
        p_unpadded = p_unpadded.reshape(-1, tokens)
        t_unpadded = torch.masked_select(tgt, nonpad_loc)
        ce_loss = super().forward(p_unpadded, t_unpadded)
        return ce_loss


class D3PMLVBLossMSA(KLDivLoss): # TODO make sure this matches seqs
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
        self.K = tokenizer.K
        super().__init__(reduction=reduction, log_target=log_target)

    def forward(self, src_one_hot, q, predictions, tgt, tgt_one_hot, input_mask, timestep, Q, Q_bar):
        p = torch.nn.functional.softmax(predictions[:, :, :, :self.K], dim=3)  # ignoring specials
        losses = []
        nonpad_loc = input_mask.sum(axis=2)
        for i in range(len(tgt)): # enumerate over batch
            D = int(nonpad_loc[i][0])  # all seq in one MSA are padded to the same length, use first seq as ref
            if timestep[i] == 1:
                # CE (L_t=0)
                reconstruction_loss = D3PMCELossMSA(tokenizer=self.tokenizer)
                r_loss = reconstruction_loss(predictions[i].unsqueeze(0), tgt[i].unsqueeze(0), input_mask[i].unsqueeze(0))
                #print(timestep[i], r_loss)
                losses.append(r_loss)
            elif timestep[i] == self.tmax:  # Not needed to compute gradients
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                q_true = q[i, :, :D, :]
                prior = sample_prior3D(q_true.shape[0], q_true.shape[1], q_true.shape[2], _len=self.tokenizer.alphabet)
                prior = prior.to(tgt.device)
                kl_loss_i = super().forward(prior.log(), q_true)  # fKLDivLoss expects input in log-space
                losses.append(kl_loss_i)
                #print(timestep[i], kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta_marg)
                pred = p[i, :, :D].flatten(start_dim=0, end_dim=1) # [pos x tokens]
                pred = pred.to(torch.float64)  # must use 64 not 32 or p_theta_marg
                x_t = src_one_hot[i, :, :D, :].flatten(start_dim=0, end_dim=1)
                x_0 = tgt_one_hot[i, :, :D, :].flatten(start_dim=0, end_dim=1)
                #print(x_t.shape, x_0.shape)

                # Calculate p_theta_marg, simplified for loops
                A = torch.mm(x_t, torch.t(Q[timestep[i]]))  # [P x K]
                B = torch.mm(x_0, Q_bar[timestep[i] - 1])  # P x K
                B_expand = Q_bar[timestep[i]-1].unsqueeze(0).expand(A.shape[0], self.K, self.K) # [ P x K x K]
                q_t = torch.mul(A.unsqueeze(1), B_expand)  # batch mul (broadcast logit dim)
                p_theta_marg = torch.bmm(q_t, pred.unsqueeze(2)).squeeze()  # this marginalizes over logits
                p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)  # renormalize probabilities at each position before KL
                p_theta_marg = p_theta_marg.to(tgt.device)

                # calculate q_t_minus_1, simplified for loops
                num = torch.mul(A, B)
                denom = torch.bmm(torch.mm(x_0, Q_bar[timestep[i]]).unsqueeze(1), x_t.unsqueeze(2))
                q_t_minus1 = num / denom.squeeze().unsqueeze(1)

                kl_loss_i = super().forward(p_theta_marg.log(), q_t_minus1)  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
                #print(timestep[i], kl_loss_i)
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        return lvb