import torch
import torch.nn as nn

class TotalLoss(nn.Module):
    def __init__(self, stop_weight=1, att_weight=0, bandwidth=50) -> None:
        super(TotalLoss, self).__init__()
        self.mel_loss = nn.L1Loss()
        self.post_mel_loss = nn.L1Loss()
        self.stop_loss = nn.BCEWithLogitsLoss()
        self.allignment_loss = DiagonalLoss(bandwidth)
        self.stop_weight = stop_weight
        self.att_weight = att_weight

    def forward(self, mel_pred, mel, postnet_pred, stop_preds, stop_tokens, attn):
        mel_loss = self.mel_loss(mel_pred, mel)
        post_mel_loss = self.post_mel_loss(postnet_pred, mel)
        stop_loss = self.stop_loss(stop_preds, stop_tokens)
        if self.att_weight != 0:
            att_loss = self.allignment_loss(attn)
        else:
            att_loss = 0
        loss = mel_loss + post_mel_loss + self.stop_weight*stop_loss + self.att_weight*att_loss
        return loss, mel_loss, post_mel_loss, stop_loss, att_loss


class DiagonalLoss(nn.Module):
    def __init__(self, bandwidth=50) -> None:
        super().__init__()
        self.bandwidth = bandwidth
   

    def forward(self, attn, T, S):
        b = self.bandwidth
        k = S // T
        attn = torch.sum(A, 0) / attn.shape[0]
        sum = 0
        for t in range(1, T-1):
            sum += torch.sum(attn[t, k*t-b:k*t+b-1], 0)
        sum /= S
        return -1*sum