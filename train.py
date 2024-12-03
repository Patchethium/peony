import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import Peony
from data import MelDataset, collate_pad
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm.auto import tqdm

def log(writer, name, value, step):
    writer.add_scalar(name, value, step)

def train():
    conf = OmegaConf.load("conf.yaml")
    ds = MelDataset(**conf.data, device=conf.train.device)
    dl = DataLoader(ds, conf.train.batch_size, collate_fn=collate_pad)
    model = Peony(**conf.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr)
    recon_w = conf.train.loss_weights.recon
    commit_w = conf.train.loss_weights.commit
    codebook_w = conf.train.loss_weights.codebook
    label_w = conf.train.loss_weights.label

    writer = SummaryWriter()
    step = 0
    for e in range(conf.train.epochs):
        for mel, codes, mask in tqdm(dl):
            step += 1
            optimizer.zero_grad()
            mel = mel.to(conf.train.device)
            codes = codes.to(conf.train.device)
            mel_hat, commitment_loss, codebook_loss, label_loss = model(mel, codes, mask)
            recon_loss = F.mse_loss(mel_hat, mel, reduction="none").mean(-1).sum() / mask.sum()

            loss = (
                recon_loss * recon_w
                + commitment_loss * commit_w
                + codebook_loss * codebook_w
                + label_loss * label_w
            )

            log(writer, "recon_loss", recon_loss, step)
            log(writer, "commitment_loss", commitment_loss, step)
            log(writer, "codebook_loss", codebook_loss, step)
            log(writer, "label_loss", label_loss, step)
            log(writer, "total_loss", loss, step)
            loss.backward()
            optimizer.step()
        else:
            print(f"Epoch {e}")

if __name__ == "__main__":
    train()