import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class Encoder(nn.Module):
    def __init__(self, n_mels, d_model, n_bottleneck, kernel_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.pre = nn.Linear(n_mels, d_model)
        self.layers = nn.ModuleList(
            [
                WNConv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.post = nn.Linear(d_model, n_bottleneck)

    def forward(self, mel):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = self.pre(mel)
        x = rearrange(x, "b n d -> b d n")
        for layer, norm in zip(self.layers, self.norms):
            x = self.dropout(F.relu(layer(x)))
            x = norm(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.post(x)
        return x
    
class Decoder(nn.Module):
    """
    Just the reverse of the encoder
    """
    def __init__(self, n_mels, d_model, n_bottleneck, kernel_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.model = Encoder(n_bottleneck, d_model, n_mels, kernel_size, num_layers, dropout) # swap the n_mels and n_bottleneck
    
    def forward(self, bottleneck):
        return self.model(bottleneck)


class VQVAE(nn.Module):
    def __init__(self, dim, n_codebook):
        super(VQVAE, self).__init__()
        self.codebook = nn.Parameter(torch.randn(n_codebook, dim))
    
    def forward(self, z, i, mask):
        """
        z: the intermediate bottleneck representation [B,T,D_bottle]
        i: the label index, [B,T]
        """
        # compute the distance between the codebook and the bottleneck
        z = F.normalize(z)
        codebook = F.normalize(self.codebook)
        code = codebook[i] # [B,T,D_bottle]
        code = F.normalize(code)

        dist = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ codebook.t()
            + codebook.pow(2).sum(-1, keepdim=True).t()
        )
        dist.masked_fill_(~mask.unsqueeze(-1), -1e9)
        dist = F.softmax(-dist, dim=-1) # [B,T,N_codebook]
        label_loss = F.cross_entropy(dist.transpose(1, 2), i, reduction="none") # [B,T]
        commitment_loss = F.mse_loss(z, code.detach(), reduction="none").mean(-1) # [B,T]
        codebook_loss = F.mse_loss(code, z.detach(), reduction="none").mean(-1) # [B,T]

        return code, commitment_loss, codebook_loss, label_loss


class RVQ(nn.Module):
    def __init__(self, dim, n_codebook, n_layers):
        super(RVQ, self).__init__()
        self.codebook = nn.Parameter(torch.randn(n_codebook, dim))
        self.quantizers = nn.ModuleList([VQVAE(dim, n_codebook) for _ in range(n_layers)])
    
    def forward(self, z, i, mask):
        """
        z: [B,T,D]
        i: [B,T,N]; N=n_layers
        mask: [B,T]
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        label_loss = 0

        for idx, quantizer in enumerate(self.quantizers):
            c, commitment, codebook, label = quantizer(z, i[:, :, idx], mask)
            z_q = z_q + c * mask.unsqueeze(-1)
            commitment_loss += (commitment * mask).sum() / mask.sum()
            codebook_loss += (codebook * mask).sum() / mask.sum()
            label_loss += (label * mask).sum() / mask.sum()
            residual = residual - c
        return z_q, commitment_loss, codebook_loss, label_loss


class Peony(nn.Module):
    def __init__(self, n_mels, d_model, n_bottleneck, n_codebook, n_quantizer, kernel_size, num_layers, dropout):
        super(Peony, self).__init__()
        self.encoder = Encoder(n_mels, d_model, n_bottleneck, kernel_size, num_layers, dropout)
        self.decoder = Decoder(n_mels, d_model, n_bottleneck, kernel_size, num_layers, dropout)
        self.rvq = RVQ(n_bottleneck, n_codebook, n_quantizer)
    
    def forward(self, mel, i, mask):
        """
        mel: [B,T,D]
        i: [B,T,N]; N=n_layers
        mask: [B,T]
        """
        z = self.encoder(mel)
        z = z * mask.unsqueeze(-1)
        _, commitment_loss, codebook_loss, label_loss = self.rvq(z, i, mask)
        mel_hat = self.decoder(z)
        return mel_hat, commitment_loss, codebook_loss, label_loss