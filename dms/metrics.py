import torch

class MaskedAccuracyMSA(object):
    """Masked accuracy.
    Inputs:
        N (batch size), L (MSA length), C (probability per output token)
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)

    Outputs: accuracy of predicted MSA
    """

    def __call__(self, pred, tgt, mask):
        nonpad_loc = mask.bool()
        pred = pred[:, :, :, :-4] # cut out extra chars not included in blosum matrix
        batchsize, length, depth, tokens = pred.shape
        masked_pred = torch.masked_select(pred, nonpad_loc.unsqueeze(-1).expand(pred.shape))
        masked_pred = masked_pred.reshape(-1, tokens)
        _, p = torch.max(torch.nn.functional.softmax(masked_pred, dim=-1), -1)
        masked_tgt = torch.masked_select(tgt, nonpad_loc)
        #print("target/pred", masked_tgt, p)
        #print("p", p.shape, p)
        accu = torch.mean((p == masked_tgt).float())
        return accu