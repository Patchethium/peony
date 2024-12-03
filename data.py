import torchaudio as ta
from torch.utils.data import Dataset
import dac
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Tuple

def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    lens = [b[1].shape[0] for b in batch]
    arange = torch.arange(0, max(lens))
    mask = arange.unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
    mels = []
    codes = []
    for mel, code in batch:
        mels.append(mel)
        codes.append(code)
    mels = pad_sequence(mels, batch_first=True)
    codes = pad_sequence(codes, batch_first=True)
    return mels, codes, mask

def lens2mask(lens: List[int]):
    arange = torch.arange(max(lens))
    lens = torch.LongTensor(lens)
    mask = arange < lens.unsqueeze(1)
    return mask


class MelDataset(Dataset):
    def __init__(self, path, n_fft, n_hop, sr, n_mels, device=None) -> None:
        super().__init__()
        self.sr = sr
        with open(path) as f:
            self.files = list(map(lambda x: x.strip(), f.readlines()))
        self.mel = ta.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=n_hop,
            sample_rate=sr,
            n_mels=n_mels,
            f_min=0,
            f_max=sr // 2,
        )
        x, origin_sr = ta.load(self.files[0])
        if origin_sr != sr:
            self.resample = ta.transforms.Resample(orig_freq=origin_sr, new_freq=sr)
        else:
            self.resample = None
        sr_str = None
        match sr:
            case 16000:
                sr_str = "16khz"
            case 24000:
                sr_str = "24khz"
            case 44100:
                sr_str = "44khz"
            case _:
                raise ValueError("Invalid sample rate")
        model_path = dac.utils.download(model_type=sr_str)
        self.dac = dac.DAC.load(model_path)
        if device is not None:
            self.dac.to(device)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, _sr = ta.load(self.files[idx])
        if self.resample is not None:
            x = self.resample(x)
        mel = self.mel(x).transpose(-2, -1).squeeze()
        x = self.dac.preprocess(x, self.sr)
        # codes; [1, n_quantizers, T]
        _, codes, _, _, _ = self.dac.encode(x.unsqueeze(0))
        codes = codes.transpose(-2, -1).squeeze()
        return mel, codes
