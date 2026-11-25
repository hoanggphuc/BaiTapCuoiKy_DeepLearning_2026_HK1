import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

def load_audio(path, backend="soundfile"): #load audio file và trả về waveform và sample_rate
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} không tồn tại.")

    waveform, sample_rate = torchaudio.load(path, normalize=True, backend=backend)
    return waveform.float(), sample_rate

def to_mono(waveform): #chuyển waveform về 1 channel bằng cách lấy trung bình các channel
    if waveform.ndim != 2:
        raise ValueError("Waveform phải có 2 chiều (channels, samples).")
    return waveform.mean(dim=0, keepdim=True)

def resample(waveform, orig_sr, target_sr): #chuyển sample_rate của waveform về 22050 Hz
    if orig_sr == target_sr: #trường hợp sample_rate = target_sr
        return waveform

    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)  

def pad_or_trim(waveform, target_length): #thêm hoặc cắt bớt samples của waveform để đạt target_length
    current_length = waveform.shape[1]

    if current_length > target_length: #trường hợp samples dài hơn target_length, cắt bớt samples
        return waveform[:, :target_length]
    
    elif current_length < target_length: #trường hợp samples ngắn hơn target_length, thêm 0 vào cuối samples
        pad_amount = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, pad_amount))
    
    else:
        return waveform

def process_audio(path, config):
    waveform, sample_rate = load_audio(path, backend=config.backend)

    if config.mono:
        waveform = to_mono(waveform)

    waveform = resample(waveform, orig_sr=sample_rate, target_sr=config.sample_rate)
    waveform = pad_or_trim(waveform, target_length=config.num_samples)

    return waveform