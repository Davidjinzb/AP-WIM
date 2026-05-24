from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
class IMAGENETSUBSET1000(TensorDataset):
    def __init__(self):
        data_tensor_path = 'data/img_batch.pt'
        target_tensor_path = 'data/label_batch.pt'
        data_tensor = torch.load(data_tensor_path)
        target_tensor = torch.load(target_tensor_path)
        super(IMAGENETSUBSET1000, self).__init__(data_tensor, target_tensor)
        

# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import scipy.io.wavfile as wavf


class ESC50(Dataset):
    """
    返回：
      Xlgmel: torch.FloatTensor [1, T, n_mels]   (log-mel in dB, time-major)
      y     : torch.LongTensor  []              (class index)
      wav_path: str
      idx   : int
    """

    def __init__(
        self,
        mode="train",
        num_FOLD=1,
        root_dir="ESC-50-master",
        select_class=None,
        add_noise=False,
        nfft=1024,
        hop=512,
        sr=44100,
        nmel=128,
    ):
        super().__init__()
        assert mode in ["train", "test"], "mode should be 'train' or 'test'"

        self.mode = mode
        self.num_FOLD = int(num_FOLD)
        self.root_dir = root_dir
        self.sr = sr
        self.nfft = nfft
        self.hop = hop
        self.nmel = nmel
        self.noise = add_noise

        if select_class is None or select_class == []:
            self.subset = list(range(50))
        else:
            self.subset = list(select_class)

        meta_path = os.path.join(self.root_dir, "meta", "esc50.csv")
        rows = []
        with open(meta_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for r in reader:
                rows.append(r)
        rows = np.array(rows)

        idx_list = []
        for i in range(1, rows.shape[0]):
            fold = int(rows[i][1])
            lab = int(rows[i][2])
            if lab not in self.subset:
                continue
            if self.mode == "test" and fold == self.num_FOLD:
                idx_list.append(i)
            elif self.mode == "train" and fold != self.num_FOLD:
                idx_list.append(i)

        self.arr = rows[idx_list]

        self._window = torch.hann_window(self.nfft)
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.nfft,
            hop_length=self.hop,
            n_mels=self.nmel,
            power=2.0,   # power mel
        )
        self._db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        samp = self.arr[idx]
        wav_path = os.path.join(self.root_dir, "audio", samp[0])

        y = int(samp[2])
        if len(self.subset) < 50:
            y = int(self.subset.index(y))
        y = torch.tensor(y, dtype=torch.long)

        # 1) load wav
        fs, audio = wavf.read(wav_path)
        audio = audio.astype(np.float32)

        # 2) normalize (safe)
        maxv = float(np.max(np.abs(audio)))
        if maxv > 1e-8:
            audio = audio / maxv
        else:
            audio = np.zeros_like(audio, dtype=np.float32)

        waveform = torch.from_numpy(audio).float()  # [T]

        # 3) optional noise
        if self.noise:
            energy_signal = waveform.pow(2).mean()
            noise = torch.randn_like(waveform) * 0.05
            energy_noise = noise.pow(2).mean()
            const = torch.sqrt(energy_signal / (energy_noise + 1e-8))
            waveform = waveform + const * noise

        # 4) log-mel (dB), output [T, n_mels]
        mel = self._mel(waveform)             # [n_mels, T]
        Xlgmel = self._db(mel).transpose(0, 1)  # [T, n_mels]

        return Xlgmel.unsqueeze(0).float(), y, waveform
