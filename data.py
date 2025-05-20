import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_scp(root_dir) :
    scp_path = "wav.scp"
    wav_files = [f for f in os.listdir(root_dir) if f.lower().endswith(".wav")]
    with open(scp_path, 'w') as f:
        for wav in sorted(wav_files) :
            utt = os.path.splitext(wav)[0]
            path = os.path.abspath(os.path.join(root_dir,wav))
            f.write(f"{utt}{path}\n")

class mel_dataset(Dataset) :
    def __init__(self,sample_rate = 22050, duration = 2.0, 
                 n_mels = 80, n_fft = 1024, hop_length = 256, transform = None) :
        self.entries = []
        with open("wav.scp", "r") as f:
            for line in f:
                utt, path = line.strip().split()
                self.entries.append((utt, path))
        
        self.sample_rate = sample_rate
        #采样率
        self.duration = duration
        #样本时间长度
        self.n_mels = n_mels
        #mel系数个数
        self.n_fft = n_fft
        #窗长
        self.hop_length = hop_length
        #补偿
        self.transform = transform
        #后处理变化（不知道啥）
        self. num_samples = int(self.sample_rate * (int)(self.duration))
    
    def __len__(self) :
        return len(self.entries)
    
    def __getitem__(self, idx):
        utt, path = self.entries[idx]
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        waveform = waveform.mean(dim = 0, keep_dim = True)
        if waveform.shape[1] < self.num_samples:
            pad_size = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else :
            waveform = waveform[:, :self.num_samples]
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            n_mels = self.n_mels
        )(waveform)

        mel_db = torch.log(mel_spec + 1e-6)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
        if self.transform :
            mel_norm = self.transform(mel_norm)
        return utt, mel_norm
    

if __name__ == "__main__" :
    get_scp("C:/Users/gigggjjffd/Desktop/work/machine learning/VAE/LJSpeech-1.1/LJSpeech-1.1/wavs")
    dataset = mel_dataset()
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)