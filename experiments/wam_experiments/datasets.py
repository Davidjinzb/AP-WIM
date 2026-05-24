import os
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
class IMAGENETSUBSET1000(TensorDataset):
    def __init__(self):
        tensor_dir = os.environ.get("APWIM_IMAGENET_TENSOR_DIR", "data")
        data_tensor_path = os.path.join(tensor_dir, 'img_batch.pt')
        target_tensor_path = os.path.join(tensor_dir, 'label_batch.pt')
        data_tensor = torch.load(data_tensor_path)
        target_tensor = torch.load(target_tensor_path)
        super(IMAGENETSUBSET1000, self).__init__(data_tensor, target_tensor)
        
class CUBDATASET(ImageFolder):
    def __init__(self, root="data/CUB/test", transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        super(CUBDATASET, self).__init__(root=root, transform=transform)
        

# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
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
        root_dir=None,
        select_class=None,
        add_noise=False,
        nfft=1024,
        hop=512,
        sr=44100,
        nmel=128,
    ):
        super().__init__()
        import torchaudio

        assert mode in ["train", "test"], "mode should be 'train' or 'test'"

        self.mode = mode
        self.num_FOLD = int(num_FOLD)
        self.root_dir = root_dir or os.environ.get("APWIM_ESC50_ROOT", "data/ESC-50-master")
        self.sr = sr
        self.nfft = nfft
        self.hop = hop
        self.nmel = nmel
        self.noise = add_noise

        # 选择类别
        if select_class is None or select_class == []:
            self.subset = list(range(50))
        else:
            self.subset = list(select_class)

        # 读 meta
        meta_path = os.path.join(self.root_dir, "meta", "esc50.csv")
        rows = []
        with open(meta_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for r in reader:
                rows.append(r)
        rows = np.array(rows)  # 含表头

        # 过滤 split
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

        self.arr = rows[idx_list]  # 不含表头

        # 预建 torchaudio 变换（避免每次 __getitem__ new）
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



# -*- coding: utf-8 -*-
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FMA_SMALL(Dataset):
    """
    FMA-small (8000 tracks, 8 genres) 音乐分类
    ✅ 固定长度，避免 DataLoader stack 报错

    返回格式（对齐你贴的 ESC50 实现）：
      Xlgmel: [1, T, n_mels]
      y     : torch.long 标量
      waveform: [T]
    """

    def __init__(
        self,
        mode="train",                 # 'train' or 'test'
        root_dir="data_fma",          # 包含 fma_small/ 和 fma_metadata/
        test_ratio=0.2,
        seed=42,
        add_noise=False,
        nfft=1024,
        hop=512,
        sr=22050,
        nmel=128,
        clip_seconds=5.0,             # ✅ 固定长度
        use_ffmpeg_backend=True,      # mp3 读取建议 True
    ):
        super().__init__()
        import torchaudio

        self._torchaudio = torchaudio
        assert mode in ["train", "test"]
        self.mode = mode
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "fma_small")
        self.meta_path = os.path.join(root_dir, "fma_metadata", "tracks.csv")

        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"找不到音频目录: {self.audio_dir}（请确认已解压 fma_small.zip）")
        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(f"找不到元数据: {self.meta_path}（请确认已解压 fma_metadata.zip）")

        if use_ffmpeg_backend:
            try:
                self._torchaudio.set_audio_backend("ffmpeg")
            except Exception:
                pass

        self.noise = add_noise
        self.nfft = nfft
        self.hop = hop
        self.sr = sr
        self.nmel = nmel

        self.clip_len = int(round(self.sr * float(clip_seconds)))
        if self.clip_len <= 0:
            raise ValueError("clip_seconds 必须 > 0")

        # 预建变换（跟 ESC50 一样避免每次 new）
        self._mel = self._torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.nfft,
            hop_length=self.hop,
            n_mels=self.nmel,
            power=2.0,
        )
        self._db = self._torchaudio.transforms.AmplitudeToDB(stype="power")

        # 1) 从 tracks.csv 读 track_id -> genre_top
        with open(self.meta_path, "r", newline="") as f:
            reader = csv.reader(f)
            header1 = next(reader)
            header2 = next(reader)
            headers = []
            for a, b in zip(header1, header2):
                a = a.strip()
                b = b.strip()
                headers.append(f"{a}.{b}" if a else b)

            try:
                genre_col = headers.index("track.genre_top")
            except ValueError:
                genre_col = None
                for i, h in enumerate(headers):
                    if "genre_top" in h:
                        genre_col = i
                        break
                if genre_col is None:
                    raise RuntimeError("tracks.csv 里没找到 genre_top 列")

            items = []
            genres = set()
            for row in reader:
                if not row or len(row) <= genre_col:
                    continue
                track_id = row[0].strip()
                genre = row[genre_col].strip()
                if genre == "" or genre.lower() == "nan":
                    continue

                tid_int = int(track_id)
                fname = f"{tid_int:06d}.mp3"
                subdir = fname[:3]
                path = os.path.join(self.audio_dir, subdir, fname)
                if os.path.isfile(path):
                    items.append((path, genre))
                    genres.add(genre)

        if len(items) == 0:
            raise RuntimeError("没找到任何音频文件（请确认结构是 fma_small/000/xxxxxx.mp3）")

        self.genre_list = sorted(list(genres))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}

        # 2) 按类均衡切分 train/test
        per_class = {g: [] for g in self.genre_list}
        for i, (p, gname) in enumerate(items):
            per_class[gname].append(i)

        g = torch.Generator().manual_seed(seed)
        train_ids, test_ids = [], []
        for gname, idxs in per_class.items():
            idxs_t = torch.tensor(idxs)
            perm = idxs_t[torch.randperm(len(idxs_t), generator=g)].tolist()
            k = int(round(len(perm) * test_ratio))
            test_ids.extend(perm[:k])
            train_ids.extend(perm[k:])

        self.items = items
        self.indices = train_ids if self.mode == "train" else test_ids

    def __len__(self):
        return len(self.indices)

    def _fix_length(self, waveform_1d: torch.Tensor) -> torch.Tensor:
        """把 waveform 固定到 self.clip_len（train 随机裁剪；test 中心裁剪；不足补零）"""
        L = waveform_1d.numel()
        if L >= self.clip_len:
            if self.mode == "train":
                start = torch.randint(0, L - self.clip_len + 1, (1,)).item()
            else:
                start = (L - self.clip_len) // 2
            return waveform_1d[start:start + self.clip_len]
        else:
            pad = self.clip_len - L
            return F.pad(waveform_1d, (0, pad))

    def __getitem__(self, idx):
        base_i = self.indices[idx]
        wav_path, genre = self.items[base_i]
        y = torch.tensor(self.genre_to_idx[genre], dtype=torch.long)

        waveform, sample_rate = self._torchaudio.load(wav_path)  # [C,T]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [T]

        # resample
        if sample_rate != self.sr:
            waveform = self._torchaudio.functional.resample(waveform, sample_rate, self.sr)

        # safe normalize
        maxv = waveform.abs().max()
        if maxv > 1e-8:
            waveform = waveform / maxv
        else:
            waveform = torch.zeros_like(waveform)

        # ✅ 固定长度
        waveform = self._fix_length(waveform)

        # optional noise（跟 ESC50 风格一致）
        if self.noise:
            energy_signal = waveform.pow(2).mean()
            noise = torch.randn_like(waveform) * 0.05
            energy_noise = noise.pow(2).mean()
            const = torch.sqrt(energy_signal / (energy_noise + 1e-8))
            waveform = waveform + const * noise

        # log-mel (dB): [T, n_mels]
        mel = self._mel(waveform)              # [n_mels, T]
        Xlgmel = self._db(mel).transpose(0, 1) # [T, n_mels]

        # ✅ 对齐 ESC50: return (Xlgmel, y, waveform)
        return Xlgmel.unsqueeze(0).float(), y, waveform


# datasets_audiomnist.py
# AudioMNIST loader: return exactly like ESC50
# (Xlgmel[1,T,n_mels], y(long), waveform[T])

import os
import re
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class AudioMNIST(Dataset):
    """
    Directory layout (your current):
      root_dir/
        train/01/*.wav
        valid/01/*.wav
        test/01/*.wav
        ...
        train/60/*.wav

    Filename pattern (common AudioMNIST):
      {digit}_{speaker}_{utterance}.wav
      e.g. 0_01_0.wav  (digit=0, speaker=01)

    Return (align with ESC50):
      Xlgmel: torch.FloatTensor [1, T, n_mels]   (log-mel dB, time-major)
      y     : torch.LongTensor  []              (digit 0-9)
      waveform: torch.FloatTensor [T]           (mono, fixed length)
    """

    def __init__(
        self,
        mode="train",                         # train/valid/test
        root_dir="data/audiomnist",
        sr=16000,
        nfft=1024,
        hop=512,
        nmel=128,
        clip_seconds=1.0,                     # AudioMNIST is ~1s; keep fixed for DataLoader
        add_noise=False,
        use_ffmpeg_backend=False,             # wav usually doesn't need ffmpeg
        speakers=None,                        # None or list[int] e.g. [1,2,3]
    ):
        super().__init__()
        import torchaudio

        self._torchaudio = torchaudio
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, mode)

        self.sr = int(sr)
        self.nfft = int(nfft)
        self.hop = int(hop)
        self.nmel = int(nmel)
        self.noise = bool(add_noise)

        self.clip_len = int(round(self.sr * float(clip_seconds)))
        if self.clip_len <= 0:
            raise ValueError("clip_seconds must be > 0")

        if use_ffmpeg_backend:
            try:
                self._torchaudio.set_audio_backend("ffmpeg")
            except Exception:
                pass

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        # build file list
        wavs = []
        # speaker folders are like "01", "02", ...
        speaker_dirs = sorted([d for d in glob.glob(os.path.join(self.split_dir, "*")) if os.path.isdir(d)])
        if speakers is not None:
            sp_set = {f"{int(s):02d}" for s in speakers}
            speaker_dirs = [d for d in speaker_dirs if os.path.basename(d) in sp_set]

        for sd in speaker_dirs:
            wavs.extend(sorted(glob.glob(os.path.join(sd, "*.wav"))))

        if len(wavs) == 0:
            raise RuntimeError(f"No wav files found under {self.split_dir}")

        self.wav_paths = wavs

        # prebuild transforms (like ESC50)
        self._window = torch.hann_window(self.nfft)
        self._mel = self._torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.nfft,
            hop_length=self.hop,
            n_mels=self.nmel,
            power=2.0,
        )
        self._db = self._torchaudio.transforms.AmplitudeToDB(stype="power")

        # filename regex: digit_speaker_utt.wav
        self._pat = re.compile(r"^(\d+)_([0-9]+)_([0-9]+)\.wav$")

    def __len__(self):
        return len(self.wav_paths)

    def _fix_length(self, waveform_1d: torch.Tensor) -> torch.Tensor:
        """Fixed length to clip_len. train/valid: random crop; test: center crop; pad if needed."""
        L = waveform_1d.numel()
        if L >= self.clip_len:
            if self.mode in ["train", "valid"]:
                start = torch.randint(0, L - self.clip_len + 1, (1,)).item()
            else:
                start = (L - self.clip_len) // 2
            return waveform_1d[start:start + self.clip_len]
        else:
            return F.pad(waveform_1d, (0, self.clip_len - L))

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        base = os.path.basename(wav_path)

        m = self._pat.match(base)
        if m is None:
            # fallback: try first token as digit
            # e.g. "0-01-0.wav" etc. If your naming differs, edit here.
            digit = int(re.split(r"[_\-]", base)[0])
        else:
            digit = int(m.group(1))

        y = torch.tensor(digit, dtype=torch.long)

        waveform, sample_rate = self._torchaudio.load(wav_path)  # [C, T]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).float()             # [T]

        # resample
        if int(sample_rate) != self.sr:
            waveform = self._torchaudio.functional.resample(waveform, int(sample_rate), self.sr)

        # safe normalize
        maxv = waveform.abs().max()
        if maxv > 1e-8:
            waveform = waveform / maxv
        else:
            waveform = torch.zeros_like(waveform)

        # optional noise (same style as your ESC50)
        if self.noise:
            energy_signal = waveform.pow(2).mean()
            noise = torch.randn_like(waveform) * 0.05
            energy_noise = noise.pow(2).mean()
            const = torch.sqrt(energy_signal / (energy_noise + 1e-8))
            waveform = waveform + const * noise

        # fix length for batching
        waveform = self._fix_length(waveform)

        # log-mel(dB): [n_mels, T] -> [T, n_mels]
        mel = self._mel(waveform)                 # [n_mels, T]
        Xlgmel = self._db(mel).transpose(0, 1)    # [T, n_mels]

        return Xlgmel.unsqueeze(0).float(), y, waveform
