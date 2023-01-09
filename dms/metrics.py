import torch

class MaskedAccuracyMSA(object):
    """Masked accuracy.
    Inputs:
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)
    """

    def __call__(self, pred, tgt, mask):
        nonpad_loc = mask.bool()
        pred = pred[:, :, :, :-4] # TODO cut out extra chars
        batchsize, length, depth, tokens = pred.shape
        masked_pred = torch.masked_select(pred, nonpad_loc.unsqueeze(-1).expand(pred.shape))
        masked_pred = masked_pred.reshape(-1, tokens)
        _, p = torch.max(torch.nn.functional.softmax(masked_pred, dim=-1), -1)
        masked_tgt = torch.masked_select(tgt, nonpad_loc)
        #print("target", masked_tgt.shape, masked_tgt)
        #print("p", p.shape, p)
        accu = torch.mean((p == masked_tgt).float())
        return accu